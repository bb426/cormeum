# =============================================================================
# 
# 1-2. Categorizing
# 
# =============================================================================

__mapping__ = {
    'A': 2, # AF rhythm
    'O': 2, # Other rhythm
    '~': 2, # Noise rhythm
    'N': 0 # Normal rhythm
}

def format_labels(labels):
    return [__mapping__[x] for x in labels]
	
y = format_labels(y)
