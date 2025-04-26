import re

# Read the file content
with open('/Users/mfine/Exa/rank-test/src/rank_test/dataset.py', 'r') as f:
    content = f.read()

# Add the print statement after the batch_indices creation
pattern = r'(batch_indices = \[\s+indices\[i:i\+self\.batch_size\]\s+for i in range\(0, num_items, self\.batch_size\)\s+\])'
replacement = r'\1\n        \n        print(f"Created {len(batch_indices)} batches")'

# Replace in the content
new_content = re.sub(pattern, replacement, content)

# Write back the modified content
with open('/Users/mfine/Exa/rank-test/src/rank_test/dataset.py', 'w') as f:
    f.write(new_content)

print("Patch applied.")