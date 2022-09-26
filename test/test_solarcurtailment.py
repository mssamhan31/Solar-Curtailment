import solarcurtailment

#file_path = r"/Users/samhan/Downloads/data" #this is for running in Samhan's laptop
file_path = r"C:\Users\samha\Documents\CANVAS\data" #for running in TETB CEEM09 computer

for i in [1, 11, 14, 4, 5, 8, 9]: 
    sample_number = i
    print('Analyzing sample number {}'.format(i))
    data_file = '/data_sample_{}.csv'.format(sample_number)
    ghi_file = '/ghi_sample_{}.csv'.format(sample_number)

    solarcurtailment.compute(file_path, data_file, ghi_file)