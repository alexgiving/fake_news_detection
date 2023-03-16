n_epoches = 25

device_name = 'mps' # cuda, cpu, mps
batch_size = 15
fake_path = './dataset/Fake.csv'
true_path = './dataset/True.csv'
cache_folder = './cache/'

milestones = [5, 10, 15, 20]
test_size = 0.05

is_half_precision = False # Only for cuda. Not tested yet
