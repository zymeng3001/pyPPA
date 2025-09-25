from fabric import Connection
from pathlib import Path
import time
import sys

# REMOTE_FILE = "/home/xinting/hello_fabric.txt"
# LOCAL_FILE = Path("hello_fabric_fetched.txt")

REMOTE_FILE = "/home/xinting/train_results.json"
LOCAL_FILE = Path("fetched.json")

def run_hello(host: str, user: str = "ubuntu", key_filename: str = None):
    """
    Connect to a host via SSH, write a hello file remotely, and fetch it back.
    """
    connect_kwargs = {}
    if key_filename:
        connect_kwargs["key_filename"] = key_filename

    # open connection
    with Connection(host=host, user=user, connect_kwargs=connect_kwargs) as c:
        msg = f"Hello from {host} at {time.ctime()}"
        # run a python one-liner remotely
        # c.run(f'python3 -c "open(\'{REMOTE_FILE}\',\'w\').write(\'{msg}\\n\')"')
        # start a timer
        print(f"[{host}] starting remote job...")
        start = time.time()
        # upload local fake_train_job.py to remote user's home so remote can run it
        local_script = "/home/xinting/yimeng_repo/pyPPA/nsga2_search/fake_train_job.py"
        remote_script = "/home/xinting/fake_train_job.py"
        try:
            print(f"[{host}] uploading {local_script} -> {remote_script}...")
            c.put(local_script, remote_script)
            c.run(f'chmod 755 {remote_script}')
        except Exception as e:
            print(f"Error uploading script: {e}")
            return
        # run the uploaded script
        c.run(f'python3 {remote_script}')
        # fetch it back
        print(f"Time elapsed: {time.time() - start:.2f} seconds")
        print(f"[{host}] fetching result file...")
        try:
            c.get(REMOTE_FILE, str(LOCAL_FILE), preserve_mode=False)
        except Exception as e:
            print(f"Error fetching file: {e}")
            return
        print(f"[{host}] wrote & fetched {LOCAL_FILE}")
        print("Contents:", LOCAL_FILE.read_text().strip())

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python hello_fabric.py <internal_ip> [username] [ssh_key]")
    #     sys.exit(1)

    host = "34.136.139.51"
    user = "xinting"
    key_filename = "/home/xinting/.ssh/id_rsa"

    run_hello(host, user, key_filename)

