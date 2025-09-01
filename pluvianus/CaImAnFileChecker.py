import h5py
class CaImAnFileChecker:
    def __init__(self):
        self.file_is_online = False
        self.filename = ''
        self.likely_correct = False

    def check_file(self, filename):
        def look_for_t_online(name):
            if name == "t_online":
                self.file_is_online = True
        
        def look_for_estimates(name):
            if name == "estimates":
                self.likely_correct = True
        
        self.likely_correct = False #this is a placeholder, you can implement more checks if needed                
        self.file_is_online = False
        self.filename = filename
        with h5py.File(filename, 'r') as f:
            f.visit(look_for_estimates) # set self.likely_correct to True if "estimates" is found
            if not self.likely_correct:
                print('Warning: File does not contain "estimates" dataset, it may not be a valid CaImAn file.')
                return
            f.visit(look_for_t_online) # set self.file_is_online to True if "t_online" is found


    @property
    def is_online(self):
        return self.file_is_online

    @property
    def file_name(self):
        return self.filename

    @property
    def is_likely_correct(self):
        return self.likely_correct  