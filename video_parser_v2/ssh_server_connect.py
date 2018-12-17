import subprocess
import socket
import getpass

user = getpass.getuser()

command = "squeue -u "+user+" --format=\"%.30R\""

out = subprocess.check_output(command, shell=True).decode("utf-8")
out = out.split("\n")
out = [line.strip() for line in out]
out = out[1:]
server_names = [x for x in out if x is not '']

my_name = socket.gethostname()
my_name = my_name.split(".")
my_name = my_name[0]

if my_name in server_names:
    server_names.remove(my_name)

print(server_names)

i = 0
for server_name in server_names:
    first_gpu = 9000 + i
    second_gpu = 9100 + i
    print("ssh -N -f -L  "+str(first_gpu)+":"+server_name+":8123 "+user+"@"+server_name+".pvt.bridges.psc.edu")
    #print("ssh -N -f -L  "+str(second_gpu)+":"+server_name+":8666 "+user+"@"+server_name+".pvt.bridges.psc.edu")
    i+=1

i = 0
for server_name in server_names:
    N_servers = 9
    for Nth_gpu in range(N_servers):
        local_port = 9000 + Nth_gpu*11
        distant_port = 8000 + Nth_gpu*11
        print("ssh -N -f -L  "+str(local_port)+":"+server_name+":"+str(distant_port)+" "+user+"@"+server_name+".pvt.bridges.psc.edu")
    i+=1