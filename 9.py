class CYKParser:
    def __init__(self):
        self.grammar = {}
        self.non_terminals = set()
        
    def add_rule(self, lhs, rhs):
        if lhs not in self.grammar:
            self.grammar[lhs] = []
        self.grammar[lhs].append(rhs)
        self.non_terminals.add(lhs)
        
    def parse(self, input_string):
        n = len(input_string)
        table = [[set() for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for lhs, rules in self.grammar.items():
                for rhs in rules:
                    if len(rhs) == 1 and rhs[0] == input_string[i]:
                        table[i][i].add(lhs)
        
        for length in range(2, n + 1):
            for start in range(n - length + 1):
                end = start + length - 1
                
                for split in range(start, end):
                    for lhs, rules in self.grammar.items():
                        for rhs in rules:
                            if len(rhs) == 2:
                                B, C = rhs
                                if B in table[start][split] and C in table[split + 1][end]:
                                    table[start][end].add(lhs)
        
        return 'S' in table[0][n-1]

if __name__ == "__main__":
    parser = CYKParser()
    
    # Example grammar for balanced parentheses
    parser.add_rule('S', ['L', 'R'])
    parser.add_rule('S', ['S', 'S'])
    parser.add_rule('L', ['('])
    parser.add_rule('R', [')'])
    
    test_strings = ['()', '((()))', '())', '((()']
    
    for s in test_strings:
        result = parser.parse(list(s))
        print(f"String '{s}' {'is' if result else 'is not'} in the language")
