from flask import Flask, render_template, request, jsonify
from Crypto.Cipher import DES, AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256, HMAC
from Crypto.Util.number import bytes_to_long, long_to_bytes
import random
import string
import numpy as np
from numpy.linalg import inv
import random
import string
import numpy as np

app = Flask(__name__)

# Caesar Cipher
def caesar_cipher(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            shift_amount = shift % 26
            if char.islower():
                result += chr(((ord(char) - ord('a') + shift_amount) % 26) + ord('a'))
            else:
                result += chr(((ord(char) - ord('A') + shift_amount) % 26) + ord('A'))
        else:
            result += char
    return result

# Playfair Cipher
def playfair_cipher(text, key):
    # Implementation of Playfair Cipher (placeholder)
    return "Playfair Cipher not implemented yet"

# Hill Cipher
def hill_cipher(text, key_matrix):
    # Implementation of Hill Cipher (placeholder)
    return "Hill Cipher not implemented yet"

# Vigenère Cipher
def vigenere_cipher(text, key):
    result = ""
    key_length = len(key)
    for i, char in enumerate(text):
        if char.isalpha():
            shift = ord(key[i % key_length].lower()) - ord('a')
            result += caesar_cipher(char, shift)
        else:
            result += char
    return result

# Rail Fence Cipher
def rail_fence_cipher(text, rails):
    # Implementation of Rail Fence Cipher (placeholder)
    return "Rail Fence Cipher not implemented yet"

# Pseudorandom Number Generation (PNG)
def pseudorandom_number_generation(seed, length):
    random.seed(seed)
    return [random.randint(0, 255) for _ in range(length)]

# Data Encryption Standard (DES)
def des_encrypt(plaintext, key):
    cipher = DES.new(key, DES.MODE_ECB)
    padded_text = pad(plaintext.encode(), 8)
    return cipher.encrypt(padded_text).hex()

def des_decrypt(ciphertext, key):
    cipher = DES.new(key, DES.MODE_ECB)
    decrypted = cipher.decrypt(bytes.fromhex(ciphertext))
    return unpad(decrypted, 8).decode()

# Advanced Encryption Standard (AES)
def aes_encrypt(plaintext, key):
    cipher = AES.new(key, AES.MODE_ECB)
    padded_text = pad(plaintext.encode(), 16)
    return cipher.encrypt(padded_text).hex()

def aes_decrypt(ciphertext, key):
    cipher = AES.new(key, AES.MODE_ECB)
    decrypted = cipher.decrypt(bytes.fromhex(ciphertext))
    return unpad(decrypted, 16).decode()

# RSA Encryption Algorithm
def rsa_encrypt(plaintext, public_key):
    cipher = PKCS1_OAEP.new(public_key)
    return cipher.encrypt(plaintext.encode()).hex()

def rsa_decrypt(ciphertext, private_key):
    cipher = PKCS1_OAEP.new(private_key)
    return cipher.decrypt(bytes.fromhex(ciphertext)).decode()

# Diffie-Hellman Key Exchange Algorithm
def diffie_hellman_key_exchange(p, g, private_a, private_b):
    public_a = pow(g, private_a, p)
    public_b = pow(g, private_b, p)
    shared_secret_a = pow(public_b, private_a, p)
    shared_secret_b = pow(public_a, private_b, p)
    return shared_secret_a == shared_secret_b, shared_secret_a

# Elliptic Curve Cryptography (ECC)
def ecc_key_exchange():
    # Placeholder for ECC
    return "ECC not implemented yet"

# ElGamal Algorithm
def elgamal_encrypt(plaintext, public_key):
    # Placeholder for ElGamal
    return "ElGamal not implemented yet"

# Message Authentication Code (MAC)
def generate_mac(message, key):
    hmac = HMAC.new(key.encode(), digestmod=SHA256)
    hmac.update(message.encode())
    return hmac.hexdigest()

# Hash Algorithm
def hash_algorithm(message):
    sha256 = SHA256.new()
    sha256.update(message.encode())
    return sha256.hexdigest()

# Rail Fence Cipher
def rail_fence_encrypt(text, rails):
    fence = [[] for _ in range(rails)]
    rail = 0
    direction = 1
    for char in text:
        fence[rail].append(char)
        rail += direction
        if rail == rails - 1 or rail == 0:
            direction *= -1
    return ''.join([''.join(row) for row in fence])

def rail_fence_decrypt(ciphertext, rails):
    length = len(ciphertext)
    fence = [[] for _ in range(rails)]
    index = 0
    for rail in range(rails):
        step = 2 * (rails - 1)
        for i in range(rail, length, step):
            fence[rail].append(ciphertext[i])
            if rail != 0 and rail != rails - 1 and i + step - 2 * rail < length:
                fence[rail].append(ciphertext[i + step - 2 * rail])
    return ''.join([''.join(row) for row in fence])

# Playfair Cipher
def playfair_prepare_key(key):
    key = key.replace('j', 'i') + string.ascii_lowercase.replace('j', '')
    key_matrix = []
    for char in key:
        if char not in key_matrix and char in string.ascii_lowercase:
            key_matrix.append(char)
    return [key_matrix[i:i+5] for i in range(0, 25, 5)]

def playfair_encrypt(text, key):
    key_matrix = playfair_prepare_key(key)
    text = text.replace('j', 'i').lower()
    text = ''.join([c for c in text if c in string.ascii_lowercase])
    if len(text) % 2 != 0:
        text += 'x'
    pairs = [text[i:i+2] for i in range(0, len(text), 2)]
    ciphertext = ''
    for pair in pairs:
        row1, col1 = divmod(key_matrix.index(pair[0]), 5)
        row2, col2 = divmod(key_matrix.index(pair[1]), 5)
        if row1 == row2:
            ciphertext += key_matrix[row1][(col1 + 1) % 5] + key_matrix[row2][(col2 + 1) % 5]
        elif col1 == col2:
            ciphertext += key_matrix[(row1 + 1) % 5][col1] + key_matrix[(row2 + 1) % 5][col2]
        else:
            ciphertext += key_matrix[row1][col2] + key_matrix[row2][col1]
    return ciphertext

# Hill Cipher
def hill_encrypt(text, key_matrix):
    text = text.lower().replace(' ', '')
    n = len(key_matrix)
    text += 'x' * (n - len(text) % n) if len(text) % n != 0 else ''
    numbers = [ord(c) - ord('a') for c in text]
    encrypted = []
    key = np.array(key_matrix)
    for i in range(0, len(numbers), n):
        block = np.array(numbers[i:i+n])
        encrypted_block = np.dot(key, block) % 26
        encrypted.extend(encrypted_block.tolist())
    return ''.join([chr(num + ord('a')) for num in encrypted])

# Pseudorandom Number Generation (PNG)
def pseudorandom_generator(seed, length):
    random.seed(seed)
    return [random.randint(0, 255) for _ in range(length)]

# Elliptic Curve Cryptography (ECC)
class EllipticCurve:
    def __init__(self, a, b, p):
        self.a = a
        self.b = b
        self.p = p

class ECCKeyExchange:
    def __init__(self, curve):
        self.curve = curve

    def generate_keypair(self):
        private_key = random.randint(1, self.curve.p - 1)
        public_key = self.scalar_mult(private_key, (self.curve.a, self.curve.b))
        return private_key, public_key

    def scalar_mult(self, k, point):
        result = None
        for _ in range(k):
            result = self.point_add(result, point)
        return result

    def point_add(self, p1, p2):
        if p1 is None:
            return p2
        if p2 is None:
            return p1
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2 and y1 != y2:
            return None
        if x1 == x2:
            m = (3 * x1 * x1 + self.curve.a) * pow(2 * y1, -1, self.curve.p) % self.curve.p
        else:
            m = (y2 - y1) * pow(x2 - x1, -1, self.curve.p) % self.curve.p
        x3 = (m * m - x1 - x2) % self.curve.p
        y3 = (m * (x1 - x3) - y1) % self.curve.p
        return (x3, y3)

# ElGamal Algorithm
def elgamal_encrypt(plaintext, p, g, public_key):
    k = random.randint(2, p - 2)
    c1 = pow(g, k, p)
    s = pow(public_key, k, p)
    c2 = (plaintext * s) % p
    return (c1, c2)

def elgamal_decrypt(c1, c2, private_key, p):
    s = pow(c1, private_key, p)
    plaintext = (c2 * pow(s, -1, p)) % p
    return plaintext

# ========== New Routes ========== #
@app.route('/railfence', methods=['POST'])
def railfence():
    data = request.json
    text = data['text']
    rails = int(data['rails'])
    ciphertext = rail_fence_encrypt(text, rails)
    decrypted = rail_fence_decrypt(ciphertext, rails)
    return jsonify({'ciphertext': ciphertext, 'decrypted': decrypted})

@app.route('/playfair', methods=['POST'])
def playfair():
    data = request.json
    text = data['text']
    key = data['key']
    ciphertext = playfair_encrypt(text, key)
    return jsonify({'ciphertext': ciphertext})

@app.route('/hill', methods=['POST'])
def hill():
    data = request.json
    text = data['text']
    key_matrix = data['key_matrix']
    ciphertext = hill_encrypt(text, key_matrix)
    return jsonify({'ciphertext': ciphertext})

@app.route('/png', methods=['POST'])
def png():
    data = request.json
    seed = data['seed']
    length = int(data['length'])
    numbers = pseudorandom_generator(seed, length)
    return jsonify({'numbers': numbers})

@app.route('/ecc', methods=['POST'])
def ecc():
    data = request.json
    curve = EllipticCurve(a=data['a'], b=data['b'], p=data['p'])
    ecc = ECCKeyExchange(curve)
    private_a, public_a = ecc.generate_keypair()
    private_b, public_b = ecc.generate_keypair()
    shared_a = ecc.scalar_mult(private_a, public_b)
    shared_b = ecc.scalar_mult(private_b, public_a)
    return jsonify({
        'shared_a': shared_a,
        'shared_b': shared_b
    })

@app.route('/elgamal', methods=['POST'])
def elgamal():
    data = request.json
    p = int(data['p'])
    g = int(data['g'])
    private_key = int(data['private_key'])
    public_key = pow(g, private_key, p)
    plaintext = int(data['plaintext'])
    c1, c2 = elgamal_encrypt(plaintext, p, g, public_key)
    decrypted = elgamal_decrypt(c1, c2, private_key, p)
    return jsonify({'ciphertext': (c1, c2), 'decrypted': decrypted})


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/caesar', methods=['POST'])
def caesar():
    data = request.json
    text = data['text']
    shift = int(data['shift'])
    result = caesar_cipher(text, shift)
    return jsonify({'result': result})

@app.route('/vigenere', methods=['POST'])
def vigenere():
    data = request.json
    text = data['text']
    key = data['key']
    result = vigenere_cipher(text, key)
    return jsonify({'result': result})

@app.route('/des', methods=['POST'])
def des():
    data = request.json
    plaintext = data['text']
    key = get_random_bytes(8)
    ciphertext = des_encrypt(plaintext, key)
    decrypted = des_decrypt(ciphertext, key)
    return jsonify({'ciphertext': ciphertext, 'decrypted': decrypted})

@app.route('/aes', methods=['POST'])
def aes():
    data = request.json
    plaintext = data['text']
    key = get_random_bytes(16)
    ciphertext = aes_encrypt(plaintext, key)
    decrypted = aes_decrypt(ciphertext, key)
    return jsonify({'ciphertext': ciphertext, 'decrypted': decrypted})

@app.route('/rsa', methods=['POST'])
def rsa():
    data = request.json
    plaintext = data['text']
    key = RSA.generate(2048)
    public_key = key.publickey()
    private_key = key
    ciphertext = rsa_encrypt(plaintext, public_key)
    decrypted = rsa_decrypt(ciphertext, private_key)
    return jsonify({'ciphertext': ciphertext, 'decrypted': decrypted})

@app.route('/mac', methods=['POST'])
def mac():
    data = request.json
    message = data['message']
    key = data['key']
    mac_value = generate_mac(message, key)
    return jsonify({'mac': mac_value})

@app.route('/hash', methods=['POST'])
def hash():
    data = request.json
    message = data['message']
    hash_value = hash_algorithm(message)
    return jsonify({'hash': hash_value})

if __name__ == '__main__':
    app.run(port=8000,debug=True)
