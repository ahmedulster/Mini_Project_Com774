import numpy as np
import json

# Generate random features
features = np.random.random(562).tolist()

# Create JSON body
request_body = {"features": features}

# Save to file
with open('test_data.json', 'w') as f:
    json.dump(request_body, f)

print("Test data saved to test_data.json")