import urllib.request
import socket

def test_ip(ip):
    url = f"http://{ip}:5000/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as response:
            if response.getcode() == 200:
                print(f"[OK] {ip}")
                return True
    except Exception as e:
        print(f"[FAIL] {ip} - {e}")
    return False

if __name__ == "__main__":
    hostname = socket.gethostname()
    host_ip = socket.gethostbyname(hostname)
    
    ips = [
        '127.0.0.1', 
        '10.137.248.205', 
        '10.137.248.168', 
        '192.168.118.1',
        host_ip
    ]
    
    print("Testing IPs for Flask Server...")
    for ip in set(ips):
        test_ip(ip)

