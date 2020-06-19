
import os
import psutil

def kill_all_except_current():
    PROCNAME = "python.exe"
    python_processes = []
    for proc in psutil.process_iter():
        if proc.name() == PROCNAME:
            python_processes.append(proc)
            print(proc)


    # # kill individual pid
    # p = psutil.Process(pid=11552)
    # p.terminate()  #
    current_process = os.getpid()
    for proc_to_kill in python_processes:
        if proc_to_kill.pid != current_process:
            proc_to_kill.terminate()
        else: pass

kill_all_except_current()