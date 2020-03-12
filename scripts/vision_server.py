import socket
import json
#from alignment import alignment

class VisionServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
    
    def startServer(self):
        self.socket.listen()
        while True:
            try:
                conn, addr = self.socket.accept()
                with conn:
                    print('Connected by ', addr)
                    data = conn.recv(1024)
                    if not data:
                        break
                    camera_idx, all_labels, target_label = parseReceivedData(data)
                    print('Camera index   = ' + str(camera_idx))
                    print('Books on shelf = ' + str(all_labels))
                    print('Target book    = ' + target_label)
                    print('Finding location of book and positioning the robot in front of it...')
                    #res = alignment(lcc_code=target_label, camera_idx=0, all_labels=all_labels)
                    res = 1
                    if res:
                        conn.sendall(b'SUCCESS')
                    else:
                        conn.sendall(b'FAILURE')
                    print("Sent!")
            except KeyboardInterrupt:
                print('\nClosing socket...')
                self.socket.close()
                break


def parseReceivedData(data):
    data = json.loads(data)
    camera_idx = data['shelf']
    all_labels = data['labels']
    target_label = data['target_label']
    return camera_idx, all_labels, target_label


def main():
    host = 'middleton'
    port = 50000
    server = VisionServer(host, port)
    server.startServer()


if __name__ == '__main__':
    main()
        
