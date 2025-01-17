import socket
import json

labels = ['DG311 Gib.', 'BJ1499.S5 Kag.', 'QC21.3 Hal.', 'QC174.12 Bra.', 'PS3562.E353 Lee.',
          'PR4662 Eli.', 'HA29 Huf.', 'QA276 Whe.', 'QA76.73.H37 Lip.', 'QA76.62 Bir.']

target_label = 'PR4662 Eli.'

shelf = 0

to_send = {'target_label' : target_label, 'labels': labels, 'shelf': shelf}

HOST = '192.168.105.3'
PORT = 50000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(json.dumps(to_send).encode())
    data = s.recv(1024)

print('Received: ', data)