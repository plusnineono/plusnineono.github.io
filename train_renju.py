"""
train_renju.py
A script to learn/tune weights for the Renju AI.
Run this script to optimize the evaluation function parameters.
"""

import random
import re
import copy
import math
import collections
import datetime

# --- Game Constants ---
N = 15
EMPTY = 0
BLACK = 1
WHITE = 2
DEPTH = 2  # Depth 2 + VCF is efficient for training weights.

# Zobrist Table for Transposition Table Hashing
ZOBRIST_TABLE = [[[random.getrandbits(64) for _ in range(3)] for _ in range(N)] for _ in range(N)]

# --- Initial Weights (Baseline) ---
# These match the initial state of renju_hard.qmd
CURRENT_WEIGHTS = {
    'eval_len': 100,
    'eval_liveFour': 90000,
    'eval_sleepFour': 6000,
    'eval_liveThree': 4500,
    'eval_sleepThree': 3000,
    'eval_center': 50,
    'eval_openFour': 8000,
    'eval_oppF3': 2000,
    'eval_fourThree': 80000,
    'eval_doubleThree': 50000,
    'static_b_len4': 1500,
    'static_b_len': 159,
    'static_b_fours': 5797,
    'static_b_threes': 2542,
    'static_b_sleepThree': 1500,
    'static_w_len4': 2531,
    'static_w_len': 180,
    'static_w_fours': 4895,
    'static_w_threes': 1429,
    'static_w_sleepThree': 1000,
    'static_oppThrees': 600
}

# Keys that are actually used in the Python simplified evaluation.
# We restrict perturbation to these to ensure the AI behavior changes.
ACTIVE_KEYS = [
    'eval_len', 'eval_liveFour', 'eval_sleepFour', 'eval_liveThree', 'eval_sleepThree', 'eval_center', 'eval_openFour', 'eval_fourThree', 'eval_doubleThree',
    'static_b_len4', 'static_b_len', 'static_b_fours', 'static_b_threes', 'static_b_sleepThree',
    'static_w_len4', 'static_w_len', 'static_w_fours', 'static_w_threes', 'static_w_sleepThree'
]

# --- Simplified Game Logic for Training ---
# Note: This is a Python port of the JS logic for consistency.
class RenjuBoard:
    def __init__(self):
        self.board = [[EMPTY]*N for _ in range(N)]
        self.history = []
        self.hash = 0
        self.tt = {}
        self.vct_tt = {}
        self.vcf_tt = {}
        self.killers = collections.defaultdict(set) # ply -> set of moves
        self.history_scores = collections.defaultdict(int) # (r,c) -> score

    def in_b(self, r, c):
        return 0 <= r < N and 0 <= c < N

    def has_neighbor(self, r, c, dist=1):
        for dr in range(-dist, dist+1):
            for dc in range(-dist, dist+1):
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if self.in_b(nr, nc) and self.board[nr][nc] != EMPTY:
                    return True
        return False

    def get(self, r, c):
        if not self.in_b(r, c): return -1
        return self.board[r][c]

    def place(self, r, c, color):
        self.board[r][c] = color
        self.history.append((r, c, color))
        self.hash ^= ZOBRIST_TABLE[r][c][color]

    def undo(self):
        if self.history:
            r, c, color = self.history.pop()
            self.board[r][c] = EMPTY
            self.hash ^= ZOBRIST_TABLE[r][c][color]

    def get_candidates(self):
        if not self.history:
            return [(7, 7)]
        candidates = set()
        for r, c, _ in self.history:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        if self.board[nr][nc] == EMPTY:
                            candidates.add((nr, nc))
        return list(candidates)

    def max_line(self, r, c, color):
        best = 1
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for dr, dc in dirs:
            cnt = 1
            rr, cc = r + dr, c + dc
            while self.in_b(rr, cc) and self.board[rr][cc] == color:
                cnt += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while self.in_b(rr, cc) and self.board[rr][cc] == color:
                cnt += 1
                rr -= dr
                cc -= dc
            if cnt > best: best = cnt
        return best

    def line_string_around(self, r, c, color, dr, dc):
        # Returns string representation for regex matching
        # . = empty, x = color, o = opponent
        rr, cc = r - 5*dr, c - 5*dc
        while not self.in_b(rr, cc):
            rr += dr
            cc += dc
        
        s = []
        idx = -1
        current_idx = 0
        
        while self.in_b(rr, cc):
            if rr == r and cc == c:
                s.append('x')
                idx = current_idx
            elif self.board[rr][cc] == EMPTY:
                s.append('.')
            elif self.board[rr][cc] == color:
                s.append('x')
            else:
                s.append('o')
            rr += dr
            cc += dc
            current_idx += 1
        return "".join(s), idx

    def count_threats_granular(self, r, c, color):
        # Simplified threat counting using string matching
        # Strict Renju rules are complex; this approximates the JS logic
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        live_four = 0
        sleep_four = 0
        live_three = 0        
        sleep_three = 0

        # Optimized: Use string literals instead of regex for speed
        p_l4 = [".xxxx."]
        p_s4 = ["xxxx.", ".xxxx", "xxx.x", "x.xxx", "xx.xx"]
        p_l3 = [".xxx.", ".xx.x.", ".x.xx."]
        p_s3 = [".xxx", "xxx.", ".x.xx", "xx.x.", ".xx.x", "x.xx."]
        
        for dr, dc in dirs:
            s, idx = self.line_string_around(r, c, color, dr, dc)
            
            # Check Live Four
            is_l4 = False
            for p in p_l4:
                start = 0
                while True:
                    found = s.find(p, start)
                    if found == -1: break
                    if found <= idx < found + len(p):
                        is_l4 = True; break
                    start = found + 1
                if is_l4: break
            if is_l4:
                live_four += 1; continue

            # Check Sleep Four
            is_s4 = False
            for p in p_s4:
                start = 0
                while True:
                    found = s.find(p, start)
                    if found == -1: break
                    if found <= idx < found + len(p):
                        is_s4 = True; break
                    start = found + 1
                if is_s4: break
            if is_s4:
                sleep_four += 1; continue
                
            # Check Live Three
            is_l3 = False
            for p in p_l3:
                start = 0
                while True:
                    found = s.find(p, start)
                    if found == -1: break
                    if found <= idx < found + len(p):
                        is_l3 = True; break
                    start = found + 1
                if is_l3: break
            if is_l3:
                live_three += 1; continue

            # Check Sleep Three
            is_s3 = False
            for p in p_s3:
                start = 0
                while True:
                    found = s.find(p, start)
                    if found == -1: break
                    if found <= idx < found + len(p):
                        is_s3 = True; break
                    start = found + 1
                if is_s3: break
            if is_s3:
                sleep_three += 1
        
        return live_four, sleep_four, live_three, sleep_three

    def is_forbidden(self, r, c):
        # Black forbidden check: Overline, Double Four, Double Three
        self.board[r][c] = BLACK
        
        # 1. Overline
        if self.max_line(r, c, BLACK) > 5:
            self.board[r][c] = EMPTY
            return True
            
        # 2. Double Four / Double Three
        l4, s4, l3, s3 = self.count_threats_granular(r, c, BLACK)
        self.board[r][c] = EMPTY
        
        if (l4 + s4) >= 2: return True
        if l3 >= 2: return True
        return False

    def creates_open_four(self, r, c, color):
        self.board[r][c] = color
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        ok = False
        p_o4 = ".xxxx."
        for dr, dc in dirs:
            s, _ = self.line_string_around(r, c, color, dr, dc)
            if p_o4 in s:
                ok = True
                break
        self.board[r][c] = EMPTY
        return ok

    def get_vcf_response(self, r, c, color):
        """
        If placing a stone at (r,c) creates a Sleep Four (forcing move),
        return the coordinate (tr, tc) that the opponent MUST play to block it.
        Returns None if it's not a Sleep Four or if it's an immediate win (5 or Open 4).
        """
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        # Patterns for Sleep 4 where '.' is the threat
        patterns = [("xxxx.", 4), (".xxxx", 0), ("x.xxx", 1), ("xx.xx", 2), ("xxx.x", 3)]
        
        for dr, dc in dirs:
            s, idx = self.line_string_around(r, c, color, dr, dc)
            
            # If immediate win (5 or Open 4), no single response saves it (or game over)
            if "xxxxx" in s or ".xxxx." in s:
                return None 

            for pat, offset in patterns:
                start = 0
                while True:
                    found = s.find(pat, start)
                    if found == -1: break
                    if found <= idx < found + 5:
                        # Found a Sleep 4 pattern involving the new stone
                        # Calculate threat coordinate
                        # line_string_around scans from r - 5*dr, c - 5*dc
                        # The 's' string starts at that offset.
                        # The threat is at index `found + offset` in `s`.
                        # We need to map this back to board coordinates.
                        
                        # Re-calculate start of scan
                        rr, cc = r - 5*dr, c - 5*dc
                        while not self.in_b(rr, cc):
                            rr += dr; cc += dc
                        
                        # Walk to threat
                        threat_idx = found + offset
                        tr = rr + threat_idx * dr
                        tc = cc + threat_idx * dc
                        
                        if self.in_b(tr, tc) and self.board[tr][tc] == EMPTY:
                            return (tr, tc)
                    start = found + 1
        return None

    def get_live_three_threats(self, r, c, color):
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        threats = []
        # Patterns and defense offsets
        pats = [
            (".xxx.", [0, 4]),
            (".x.xx.", [0, 2, 5]),
            (".xx.x.", [0, 3, 5])
        ]
        for dr, dc in dirs:
            s, idx = self.line_string_around(r, c, color, dr, dc)
            for pat, defs in pats:
                start = 0
                while True:
                    found = s.find(pat, start)
                    if found == -1: break
                    if found <= idx < found + len(pat):
                        # Calculate board coords for defenses
                        rr, cc = r - 5*dr, c - 5*dc
                        while not self.in_b(rr, cc): rr += dr; cc += dc
                        for d in defs:
                            t_idx = found + d
                            tr, tc = rr + t_idx * dr, cc + t_idx * dc
                            if self.in_b(tr, tc) and self.board[tr][tc] == EMPTY:
                                threats.append((tr, tc))
                    start = found + 1
        return list(set(threats))

    def solve_vct(self, color, depth, weights):
        if depth <= 0: return None
        
        # VCT Transposition Table
        tt_entry = self.vct_tt.get(self.hash)
        if tt_entry and tt_entry[0] >= depth:
            return tt_entry[1]

        candidates = self.get_candidates()
        vct_candidates = []
        
        for r, c in candidates:
            if self.board[r][c] != EMPTY: continue
            if color == BLACK and self.is_forbidden(r, c): continue
            self.board[r][c] = color
            threats = self.get_live_three_threats(r, c, color)
            self.board[r][c] = EMPTY
            
            if threats:
                vct_candidates.append((r, c, threats))
        
        # Sort candidates by evaluation to find wins faster
        if weights:
            vct_candidates.sort(key=lambda x: self.evaluate_move(x[0], x[1], color, weights), reverse=True)
        
        # Sort candidates by simple heuristic (number of threats or proximity)
        # For now, just try them. In full engine, sort by eval.
        
        for r, c, threats in vct_candidates:
            win = True
            self.place(r, c, color)
            opp = WHITE if color == BLACK else BLACK
            for tr, tc in threats:
                if opp == BLACK and self.is_forbidden(tr, tc): continue
                self.place(tr, tc, opp)
                
                # After opponent blocks, check for a winning continuation
                # 1. Check for an immediate VCF win
                continuation = self.solve_vcf(color, 12)
                # 2. If no VCF, check for a deeper VCT win
                if not continuation:
                    continuation = self.solve_vct(color, depth - 1, weights)

                self.undo()
                if not continuation:
                    win = False
                    break
            self.undo()
            if win:
                self.vct_tt[self.hash] = (depth, (r, c))
                return (r, c)
        
        self.vct_tt[self.hash] = (depth, None)
        return None

    def solve_vcf(self, color, depth):
        """Recursive VCF solver. Returns winning move (r,c) or None."""
        if depth == 0: return None

        tt_entry = self.vcf_tt.get(self.hash)
        if tt_entry and tt_entry[0] >= depth:
            return tt_entry[1]
        
        candidates = self.get_candidates()
        for r, c in candidates:
            if self.board[r][c] != EMPTY: continue
            if color == BLACK and self.is_forbidden(r, c): continue
            
            # 1. Check Immediate Win
            self.place(r, c, color)
            if (color == BLACK and self.max_line(r, c, color) == 5) or \
               (color == WHITE and self.max_line(r, c, color) >= 5):
                self.undo()
                self.vcf_tt[self.hash] = (depth, (r, c))
                return (r, c)
            self.undo()
            
            # 2. Check Open Four (Unstoppable)
            if self.creates_open_four(r, c, color):
                self.vcf_tt[self.hash] = (depth, (r, c))
                return (r, c)
            
            # 3. Check Sleep Four (Forcing)
            # If we create a Sleep Four, opponent is forced to block.
            # We verify this by finding the threat point.
            resp = self.get_vcf_response(r, c, color)
            
            if resp:
                tr, tc = resp
                # Opponent MUST block at (tr, tc)
                # If opponent cannot block (e.g. forbidden for them? No, white has no forbidden. Black might),
                # but usually blocking is just placing a stone.
                
                opp_color = WHITE if color == BLACK else BLACK
                
                # Check if opponent blocking is forbidden (only if opponent is Black)
                if opp_color == BLACK and self.is_forbidden(tr, tc):
                    # Opponent cannot block -> Win
                    self.vcf_tt[self.hash] = (depth, (r, c))
                    return (r, c)
                
                self.place(r, c, color)
                self.place(tr, tc, opp_color) # Opponent blocks
                
                # Recurse
                win_continuation = self.solve_vcf(color, depth - 1)
                
                self.undo()
                self.undo()
                
                if win_continuation:
                    self.vcf_tt[self.hash] = (depth, (r, c))
                    return (r, c)
        
        self.vcf_tt[self.hash] = (depth, None)
        return None

    def immediate_winning_moves(self, color):
        wins = []
        candidates = self.get_candidates()
        for r, c in candidates:
            if self.board[r][c] != EMPTY: continue
            if color == BLACK and self.is_forbidden(r, c): continue
            self.place(r, c, color)
            l = self.max_line(r, c, color)
            self.undo()
            if (color == BLACK and l == 5) or (color == WHITE and l >= 5):
                wins.append((r, c))
        return wins

    def evaluate_move(self, r, c, color, weights):
        if not self.in_b(r, c) or self.board[r][c] != EMPTY:
            return -1e18
        
        # Forbidden check for Black
        if color == BLACK and self.is_forbidden(r, c):
            return -1e18

        # Unpack weights for speed
        w_len = weights['eval_len']
        w_l4 = weights['eval_liveFour']
        w_s4 = weights['eval_sleepFour']
        w_l3 = weights['eval_liveThree']
        w_s3 = weights['eval_sleepThree']
        w_43 = weights['eval_fourThree']
        w_d3 = weights['eval_doubleThree']
        w_center = weights['eval_center']
        w_o4 = weights['eval_openFour']

        self.board[r][c] = color
        length = self.max_line(r, c, color)
        l4, s4, l3, s3 = self.count_threats_granular(r, c, color)
        
        score = 0
        if (color == BLACK and length == 5) or (color == WHITE and length >= 5):
            score += 1e9
        
        score += w_len * length
        score += w_l4 * l4
        score += w_s4 * s4
        score += w_l3 * l3
        score += w_s3 * s3
        
        if (l4 + s4) > 0 and l3 > 0:
            score += w_43
        
        if color == WHITE and l3 >= 2:
            score += w_d3
        
        dx, dy = abs(c - 7), abs(r - 7)
        score += w_center * (7 - max(dx, dy))
        
        if self.creates_open_four(r, c, color):
            score += w_o4
            
        # Simplified: Opponent Free Threes calculation is expensive in Python loop
        # We omit it for the fast training loop or approximate it
        # score -= weights['eval_oppF3'] * opp_f3_count 
        
        self.board[r][c] = EMPTY
        return score

    def side_score(self, color, weights):
        s = 0
        # Optimization: Iterate only over occupied stones instead of full board
        for r, c, col in self.history:
            if col == color:
                length = self.max_line(r, c, color)
                l4, s4, l3, s3 = self.count_threats_granular(r, c, color)
                fours = l4 + s4
                threes = l3
                if color == BLACK:
                    s += (weights['static_b_len4'] if length >= 4 else 0)
                    s += weights['static_b_len'] * length
                    s += weights['static_b_fours'] * fours
                    s += weights['static_b_threes'] * threes
                    s += weights['static_b_sleepThree'] * s3
                else: # WHITE
                    s += (weights['static_w_len4'] if length >= 4 else 0)
                    s += weights['static_w_len'] * length
                    s += weights['static_w_fours'] * fours
                    s += weights['static_w_threes'] * threes
                    s += weights['static_w_sleepThree'] * s3
        return s

    def static_eval(self, pov, weights):
        sb = self.side_score(BLACK, weights)
        sw = self.side_score(WHITE, weights)
        base = sb - sw
        return base if pov == BLACK else -base

    def quiescence(self, color, alpha, beta, depth, weights):
        stand_pat = self.static_eval(color, weights)
        if stand_pat >= beta:
            return stand_pat
        if alpha < stand_pat:
            alpha = stand_pat
            
        if depth <= 0:
            return alpha
            
        candidates = self.get_candidates()
        qs_moves = []
        
        for r, c in candidates:
            if self.board[r][c] != EMPTY: continue
            if color == BLACK and self.is_forbidden(r, c): continue
            
            # Check immediate win
            self.place(r, c, color)
            l = self.max_line(r, c, color)
            self.undo()
            if (color == BLACK and l == 5) or (color == WHITE and l >= 5):
                return 1000000 + depth
            
            # Check Open Four (Forcing move)
            if self.creates_open_four(r, c, color):
                score = self.evaluate_move(r, c, color, weights)
                qs_moves.append((score, r, c))
                
        qs_moves.sort(key=lambda x: x[0], reverse=True)
        
        for _, r, c in qs_moves:
            self.place(r, c, color)
            opp_color = WHITE if color == BLACK else BLACK
            val = -self.quiescence(opp_color, -beta, -alpha, depth - 1, weights)
            self.undo()
            
            if val >= beta:
                return beta
            if val > alpha:
                alpha = val
                
        return alpha

    def search(self, depth, alpha, beta, color, weights, ply=0, allow_null=True):
        """A proper Negamax search with Alpha-Beta Pruning and Late Move Reduction."""
        # Transposition Table Lookup
        tt_entry = self.tt.get(self.hash)
        tt_move = None
        if tt_entry:
            tt_depth, tt_flag, tt_value, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_flag == 0: return tt_value # EXACT
                elif tt_flag == 1: alpha = max(alpha, tt_value) # LOWERBOUND
                elif tt_flag == 2: beta = min(beta, tt_value) # UPPERBOUND
                
                if alpha >= beta:
                    return tt_value

        if depth <= 0:
            return self.quiescence(color, alpha, beta, 4, weights)

        # Immediate Threat Pruning
        opp_color = WHITE if color == BLACK else BLACK
        opp_wins = self.immediate_winning_moves(opp_color)
        forced_cands = []
        
        if opp_wins:
            my_wins = self.immediate_winning_moves(color)
            if my_wins: return 1000000 + depth
            
            # Must block. Check if blocking moves are legal for me.
            for r, c in opp_wins:
                if color == BLACK and self.is_forbidden(r, c): continue
                forced_cands.append((r, c))
                
            if not forced_cands: return -900000

        # Null Move Pruning
        if allow_null and depth >= 3 and ply > 0:
            R = 2
            opp_color = WHITE if color == BLACK else BLACK
            val = -self.search(depth - 1 - R, -beta, -beta + 1, opp_color, weights, ply + 1, allow_null=False)
            if val >= beta: return val

        # 1. Try Hash Move (Lazy Generation)
        # If the best move from a previous search is good enough, we skip generating others.
        best_value = -float('inf')
        best_move = None
        alpha_orig = alpha
        has_searched_pv = False

        if tt_move and self.board[tt_move[0]][tt_move[1]] == EMPTY:
            self.place(tt_move[0], tt_move[1], color)
            opp_color = WHITE if color == BLACK else BLACK
            val = -self.search(depth - 1, -beta, -alpha, opp_color, weights, ply + 1, allow_null=True)
            self.undo()

            if val > best_value:
                best_value = val
                best_move = tt_move
                has_searched_pv = True
            if val > alpha:
                alpha = val
            if val >= beta:
                # Beta cutoff on Hash Move - huge speedup
                self.tt[self.hash] = (depth, 1, val, tt_move)
                return val

        if forced_cands:
            candidates = forced_cands
        else:
            candidates = self.get_candidates()

        if not candidates:
            return self.static_eval(color, weights)

        # Move ordering
        scored_moves = []
        for r, c in candidates:
            if (r, c) == tt_move: continue # Already searched
            
            score = self.evaluate_move(r, c, color, weights)
            
            # Heuristics for sorting
            if (r, c) in self.killers[ply]:
                score += 1e5 # Try killer moves early
            score += self.history_scores.get((r, c), 0) # History heuristic
            
            scored_moves.append((score, r, c))
        scored_moves.sort(key=lambda x: x[0], reverse=True)

        for i, (move_score, r, c) in enumerate(scored_moves):
            if move_score >= 1e7: # Found a winning move
                val = move_score + depth
                self.tt[self.hash] = (depth, 0, val, (r, c))
                return val

            # --- Late Move Reduction (LMR) & PVS ---
            R = 0
            if depth >= 3 and i > 3 and has_searched_pv:
                R = 1

            self.place(r, c, color)
            opp_color = WHITE if color == BLACK else BLACK

            if not has_searched_pv:
                # First move: Full window search
                value = -self.search(depth - 1, -beta, -alpha, opp_color, weights, ply + 1, allow_null=True)
                has_searched_pv = True
            else:
                # Subsequent moves: Null window search (Probe)
                value = -self.search(depth - 1 - R, -alpha - 1, -alpha, opp_color, weights, ply + 1, allow_null=True)
                
                # If LMR failed (move was better than expected), re-search with full depth (still null window)
                if R > 0 and value > alpha:
                    value = -self.search(depth - 1, -alpha - 1, -alpha, opp_color, weights, ply + 1, allow_null=True)
                
                # If Null window failed (move was better than alpha), re-search with full window
                if alpha < value < beta:
                    value = -self.search(depth - 1, -beta, -alpha, opp_color, weights, ply + 1, allow_null=True)

            self.undo()

            if value > best_value:
                best_value = value
                best_move = (r, c)

            if value > alpha:
                alpha = value

            # Alpha-Beta Pruning
            if alpha >= beta:
                # Update heuristics on cutoff
                self.killers[ply].add((r, c))
                self.history_scores[(r, c)] += depth * depth
                break # Beta cutoff

        # Store in Transposition Table
        # Replacement scheme: prefer deeper searches
        existing = self.tt.get(self.hash)
        if not existing or depth >= existing[0]:
            tt_flag = 0 # EXACT
            if best_value <= alpha_orig: tt_flag = 2 # UPPERBOUND
            elif best_value >= beta: tt_flag = 1 # LOWERBOUND
            
            self.tt[self.hash] = (depth, tt_flag, best_value, best_move)
            
        return best_value

def update_qmd_file(best_weights):
    """Reads renju_hard.qmd, replaces the WEIGHTS object, and writes it back."""
    qmd_path = 'renju_hard.qmd'

    # Construct the new WEIGHTS object string.
    # The indentation is important for matching the format in the .qmd file.
    # Double curly braces {{ and }} are used to escape braces in an f-string.
    new_weights_str = f"""  const WEIGHTS = {{
    win: 1e7,
    // Dynamic evaluation (evalMove)
    eval: {{
      len: {int(best_weights['eval_len'])},
      liveFour: {int(best_weights['eval_liveFour'])},
      sleepFour: {int(best_weights['eval_sleepFour'])},
      liveThree: {int(best_weights['eval_liveThree'])},
      sleepThree: {int(best_weights['eval_sleepThree'])},
      center: {int(best_weights['eval_center'])},
      openFour: {int(best_weights['eval_openFour'])},
      oppF3: {int(best_weights['eval_oppF3'])},
      fourThree: {int(best_weights['eval_fourThree'])},
      doubleThree: {int(best_weights['eval_doubleThree'])}
    }},
    // Static evaluation (sideScore)
    static: {{
      black: {{ len4: {int(best_weights['static_b_len4'])}, len: {int(best_weights['static_b_len'])}, fours: {int(best_weights['static_b_fours'])}, threes: {int(best_weights['static_b_threes'])}, sleepThree: {int(best_weights['static_b_sleepThree'])} }},
      white: {{ len4: {int(best_weights['static_w_len4'])}, len: {int(best_weights['static_w_len'])}, fours: {int(best_weights['static_w_fours'])}, threes: {int(best_weights['static_w_threes'])}, sleepThree: {int(best_weights['static_w_sleepThree'])} }},
      oppThrees: {int(best_weights['static_oppThrees'])}
    }}
  }};"""

    try:
        with open(qmd_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Pattern to find the entire WEIGHTS object block
        pattern = r"^\s*const WEIGHTS = \{[\s\S]*?\};"
        
        if not re.search(pattern, content, re.MULTILINE):
            print(f"Error: Could not find the 'const WEIGHTS = {{...}};' block in {qmd_path}.")
            return

        updated_content = re.sub(pattern, new_weights_str, content, count=1, flags=re.MULTILINE)
        
        # Update timestamp
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_content = re.sub(r'<span id="train-time">.*?</span>', f'<span id="train-time">{ts}</span>', updated_content)

        with open(qmd_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"\n--- Training Complete ---")
        print(f"Successfully updated weights in {qmd_path}")

    except FileNotFoundError:
        print(f"Error: {qmd_path} not found. Make sure it's in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_weights():
    """Loads the current weights from renju_hard.qmd to continue training."""
    print("Loading weights from renju_hard.qmd...")
    try:
        with open('renju_hard.qmd', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the WEIGHTS block
        w_match = re.search(r"const WEIGHTS = \{([\s\S]*?)\};", content)
        if not w_match:
            print("WEIGHTS object not found. Using defaults.")
            return CURRENT_WEIGHTS.copy()
        
        w_text = w_match.group(1)
        weights = CURRENT_WEIGHTS.copy()
        
        def extract(pattern, text):
            m = re.search(pattern, text)
            return int(m.group(1)) if m else None

        # 1. Eval block
        eval_match = re.search(r"eval:\s*\{([^}]*)\}", w_text)
        if eval_match:
            et = eval_match.group(1)
            for key, pat in [
                ('eval_len', r"len:\s*(\d+)"),
                ('eval_liveFour', r"liveFour:\s*(\d+)"),
                ('eval_sleepFour', r"sleepFour:\s*(\d+)"),
                ('eval_liveThree', r"liveThree:\s*(\d+)"),
                ('eval_sleepThree', r"sleepThree:\s*(\d+)"),
                ('eval_center', r"center:\s*(\d+)"),
                ('eval_openFour', r"openFour:\s*(\d+)"),
                ('eval_oppF3', r"oppF3:\s*(\d+)"),
                ('eval_fourThree', r"fourThree:\s*(\d+)"),
                ('eval_doubleThree', r"doubleThree:\s*(\d+)")
            ]:
                val = extract(pat, et)
                if val is not None: weights[key] = val

        # 2. Static Black/White
        for color, prefix in [('black', 'static_b'), ('white', 'static_w')]:
            m = re.search(f"{color}:\s*\{{([^}}]*)\}}", w_text)
            if m:
                t = m.group(1)
                for key, pat in [
                    (f'{prefix}_len4', r"len4:\s*(\d+)"),
                    (f'{prefix}_len', r"len:\s*(\d+)"),
                    (f'{prefix}_fours', r"fours:\s*(\d+)"),
                    (f'{prefix}_threes', r"threes:\s*(\d+)"),
                    (f'{prefix}_sleepThree', r"sleepThree:\s*(\d+)")
                ]:
                    val = extract(pat, t)
                    if val is not None: weights[key] = val

        val = extract(r"oppThrees:\s*(\d+)", w_text)
        if val is not None: weights['static_oppThrees'] = val
        
        return weights
    except Exception as e:
        print(f"Error parsing weights: {e}. Using defaults.")
        return CURRENT_WEIGHTS.copy()

# --- Training Loop ---

def get_opening_book():
    """Returns a list of standard Renju openings to ensure robust training."""
    # Coordinates are 0-indexed, (7,7) is center
    return [
        [(7, 7, BLACK), (7, 8, WHITE), (7, 6, BLACK)], # Horketsu (Direct)
        [(7, 7, BLACK), (8, 8, WHITE), (8, 6, BLACK)], # Kagetsu (Indirect)
        [(7, 7, BLACK), (6, 8, WHITE), (8, 6, BLACK)], # Kougetsu
        [(7, 7, BLACK), (8, 8, WHITE), (7, 9, BLACK)], # Matsugetsu
        [(7, 7, BLACK), (8, 8, WHITE), (6, 9, BLACK)], # Ryuusei
        [(7, 7, BLACK), (7, 8, WHITE), (6, 8, BLACK)], # Ungetsu
        [(7, 7, BLACK), (8, 8, WHITE), (8, 7, BLACK)], # Hokusei
        [(7, 7, BLACK), (6, 8, WHITE), (6, 6, BLACK)], # Suisei
    ]

def play_game(w1, w2, opening=None):
    # w1 plays Black, w2 plays White
    b = RenjuBoard()
    
    if opening:
        for r, c, color in opening:
            b.place(r, c, color)
        turn = WHITE if opening[-1][2] == BLACK else BLACK
    else:
        # Standard opening: Black center
        b.place(7, 7, BLACK)
        turn = WHITE
        
    passes = 0
    
    for _ in range(60): # Limit game length
        color = turn
        weights = w1 if color == BLACK else w2
        b.tt = {} # Clear TT to avoid mixing evaluations from different weights
        
        candidates = b.get_candidates()
        if not candidates:
            break
            
        # 0. Check for forced wins
        vcf_move = b.solve_vcf(color, 12)
        if vcf_move:
            b.place(vcf_move[0], vcf_move[1], color)
            return color
            
        vct_move = b.solve_vct(color, 4, weights)
        if vct_move:
            b.place(vct_move[0], vct_move[1], color)
            return color

        # Initial sort by static eval
        scored_moves = []
        immediate_win = None
        for r, c in candidates:
            score1 = b.evaluate_move(r, c, color, weights)
            if score1 >= 1e8:
                immediate_win = (r, c)
                break
            scored_moves.append((score1, r, c))
        
        if immediate_win:
            b.place(immediate_win[0], immediate_win[1], color)
            return color

        scored_moves.sort(key=lambda x: x[0], reverse=True)

        best_move = None
        
        # Iterative Deepening
        d_best_score = None
        
        for d in range(1, DEPTH + 1):
            alpha = -float('inf')
            beta = float('inf')
            
            # Aspiration Windows
            # If we have a previous score, search a narrow window around it first.
            if d > 2 and d_best_score is not None and abs(d_best_score) < 1e6:
                alpha = d_best_score - 2000
                beta = d_best_score + 2000

            current_scored_moves = []
            
            is_final_depth = (d == DEPTH)
            d_best_move = None
            d_best_score = -float('inf')
            
            # Helper to run the root search loop
            def run_root_search(alpha_w, beta_w):
                local_best_score = -float('inf')
                local_best_move = None
                local_scored = []
                curr_alpha = alpha_w
                
                for i, (_, r, c) in enumerate(scored_moves):
                    b.place(r, c, color)
                    opp_color = WHITE if color == BLACK else BLACK
                    
                    val = -float('inf')
                    if i == 0:
                        val = -b.search(d - 1, -beta_w, -curr_alpha, opp_color, weights, 1)
                    else:
                        val = -b.search(d - 1, -curr_alpha - 1, -curr_alpha, opp_color, weights, 1)
                        if val > curr_alpha and val < beta_w:
                            val = -b.search(d - 1, -beta_w, -curr_alpha, opp_color, weights, 1)
                    
                    b.undo()
                    
                    local_scored.append((val, r, c))
                    if val > local_best_score:
                        local_best_score = val
                        local_best_move = (r, c)
                    if val > curr_alpha:
                        curr_alpha = val # Update alpha for PVS
                return local_best_score, local_best_move, local_scored

            # Run search
            d_best_score, d_best_move, current_scored_moves = run_root_search(alpha, beta)
            
            # Check for Aspiration Window fail
            if d > 2 and (d_best_score <= alpha or d_best_score >= beta):
                # Re-search with full window
                d_best_score, d_best_move, current_scored_moves = run_root_search(-float('inf'), float('inf'))

            # Sort for next iteration
            if is_final_depth:
                # Add noise only at the very end
                current_scored_moves = [(s + random.uniform(-10, 10), r, c) for s, r, c in current_scored_moves]
                current_scored_moves.sort(key=lambda x: x[0], reverse=True)
                d_best_move = (current_scored_moves[0][1], current_scored_moves[0][2]) if current_scored_moves else None
            else:
                scored_moves = sorted(current_scored_moves, key=lambda x: x[0], reverse=True)
            
            if is_final_depth:
                best_move = d_best_move
        
        if best_move:
            b.place(best_move[0], best_move[1], color)
            # Check win
            l = b.max_line(best_move[0], best_move[1], color)
            if (color == BLACK and l == 5) or (color == WHITE and l >= 5):
                return color
        else:
            passes += 1
            if passes > 1: break
        
        # Decay history
        for k in b.history_scores: b.history_scores[k] = int(b.history_scores[k] * 0.875)
            
        turn = WHITE if turn == BLACK else BLACK
        
    return 0 # Draw

def train():
    print(f"Starting training (SPSA, Depth={DEPTH})...")
    current_weights = load_weights()
    
    # SPSA Parameters
    # c: Perturbation size (relative to weight value)
    # alpha: Learning rate
    c = 0.06
    alpha = 0.005
    
    opening_book = get_opening_book()

    # Attempt to use tqdm for a progress bar
    total_epochs = 200
    try:
        from tqdm import tqdm
        epochs_iter = tqdm(range(1, total_epochs + 1), desc="Training")
        use_tqdm = True
    except ImportError:
        epochs_iter = range(1, total_epochs + 1)
        use_tqdm = False
    
    # SPSA Loop
    for epoch in epochs_iter:
        # 1. Generate perturbation vector (Bernoulli +/- 1)
        delta = {k: (1 if random.random() < 0.5 else -1) for k in ACTIVE_KEYS}
        
        # 2. Create two perturbed weight sets
        w_plus = current_weights.copy()
        w_minus = current_weights.copy()
        
        for k in ACTIVE_KEYS:
            # Perturb proportional to current value to handle different scales
            perturbation = current_weights[k] * c * delta[k]
            w_plus[k] += perturbation
            w_minus[k] -= perturbation
            
        # 3. Play Match: w_plus vs w_minus
        # We play 2 games with swapped colors on a random opening
        opening = random.choice(opening_book)
        score = 0 # Positive if w_plus wins, Negative if w_minus wins
        
        # Game 1: Plus (Black) vs Minus (White)
        res = play_game(w_plus, w_minus, opening)
        if res == BLACK: score += 1
        elif res == WHITE: score -= 1
        
        # Game 2: Minus (Black) vs Plus (White)
        res = play_game(w_minus, w_plus, opening)
        if res == BLACK: score -= 1
        elif res == WHITE: score += 1
        
        msg = f"Epoch {epoch}: Score {score}"
        
        # 4. Update Weights (Gradient Approximation)
        if score != 0:
            # If Plus won (score > 0), move towards Plus.
            # If Minus won (score < 0), move towards Minus.
            # Update rule: w = w + alpha * score * delta * w
            # (The last 'w' is because our perturbation was relative c*w)
            for k in ACTIVE_KEYS:
                step = alpha * score * delta[k] * current_weights[k]
                current_weights[k] += step
            
            msg += " -> Weights updated."
            update_qmd_file(current_weights)
        else:
            msg += " -> Draw."

        if use_tqdm:
            tqdm.write(msg)
        else:
            print(msg)

    # Final save is handled in loop

if __name__ == "__main__":
    train()
