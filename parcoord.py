from subprocess import Popen, PIPE

d_list = [0.001, 0.0025, 0.0075, 0.01, 0.015, 0.025, 0.03, 0.1, 0.0015, 0.05, 0.035, 0.03]

ps = []
for d in d_list:
    ps.append(Popen(f"/opt/homebrew/bin/python3.10 inpar.py {d}", shell=True, stdout=PIPE))

for p in ps:
    print(p.communicate()[0])
