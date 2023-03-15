n_epoches = 200

device_name = 'mps' # cuda, cpu, mps
batch_size = 10
fake_path = './dataset/Fake.csv'
true_path = './dataset/True.csv'

milestones = [30, 60, 90, 120]
test_size = 0.05
