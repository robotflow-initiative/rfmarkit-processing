import datetime
import json
import logging
import time
from json import encoder
from json.decoder import JSONDecodeError
from typing import Dict


def call(pipe: str, cmd: Dict):
    """Send command to measurement server via pipe

    Args:
        pipe (str): The pipe, e.g. /tmp/measurement_pipe_in
        cmd (Dict): The command
    """
    # Open the pipe
    with open(pipe, 'w') as f:
        print(f"Unix timestamp: {datetime.datetime.utcnow().timestamp()}")
        # Dump the command into pipe
        json.dump(cmd, f)
        print(f"Unix timestamp: {datetime.datetime.utcnow().timestamp()}")
        f.close()

def gather(pipe: str) -> Dict:
    """Gather response from measurement server

    Args:
        pipe (str): The out pipe of measurement server, e.g. /tmp/measurement_pipe_out 

    Returns:
        Dict: The response
    """
    # Open the pipe
    cmd = {}
    with open(pipe, 'r') as f:
        print(f"Unix timestamp: {datetime.datetime.utcnow().timestamp()}")
        # Dump the command into pipe
        try:
            if f.readable:
                data = f.readline()
                cmd = json.loads(data)
        except JSONDecodeError:
            logging.warn(f"Wrong format")
        print(f"Unix timestamp: {datetime.datetime.utcnow().timestamp()}")
        f.close()
    return cmd

if __name__ == '__main__':
    # Start command, stop command and quit command
    start_cmd = {'type': 'start'}
    stop_cmd = {'type': 'stop'}
    quit_cmd = {'type': 'quit'}
    
    # Start measurement
    call('/tmp/measurement_pipe_in', start_cmd)
    print(gather('/tmp/measurement_pipe_out'))
    time.sleep(5)
    

    # Stop measurement
    call('/tmp/measurement_pipe_in', stop_cmd)
    print(gather('/tmp/measurement_pipe_out'))

    # Stop measurement
    call('/tmp/measurement_pipe_in', quit_cmd)
    print(gather('/tmp/measurement_pipe_out'))
