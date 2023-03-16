n_epoches = 200

device_name = 'cpu' # cuda, cpu, mps
batch_size = 10
fake_path = './dataset/Fake.csv'
true_path = './dataset/True.csv'
cache_folder = './cache/'

milestones = [30, 60, 90, 120]
test_size = 0.05

is_half_precision = False # Only for cuda. Not tested yet
