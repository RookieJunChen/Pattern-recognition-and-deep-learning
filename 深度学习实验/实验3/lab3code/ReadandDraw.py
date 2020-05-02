from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir="./mylog")

with open("logs/trainlogs_lr1e-3.txt", 'r') as file1:
    with open("logs/testlogs_lr1e-3.txt", 'r') as file2:
        i = 0
        for line1 in file1:
            line1 = line1.strip('\n')
            line1 = float(line1)
            line2 = file2.readline()
            line2 = line2.strip('\n')
            line2 = float(line2)
            i += 1
            writer.add_scalars('Trainloss/approve1', {'train': line1, 'test': line2}, i, walltime=i)
        writer.close()
