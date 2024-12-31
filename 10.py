def min_edit_distance(source, target):
    m, n = len(source), len(target)
    dp = [[i+j if i==0 or j==0 else 0 for j in range(n+1)] for i in range(m+1)]
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1] if source[i-1] == target[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

print(min_edit_distance("intention", "execution"))
print(min_edit_distance("Piece", "Peace"))
