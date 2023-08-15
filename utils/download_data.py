import requests

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
data = response.text

# Save the data to a file
with open("data/shakespeare.txt", "w") as f:
    f.write(data)
