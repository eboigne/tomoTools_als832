import os
import tifffile
import numpy as np
import shutil

class File:

    def findType(self):
        if self.extension == 'tif' or self.extension == 'tiff':
            return 'tif'
        elif self.extension == 'viv':
            return 'viv'
        else:
            print('File extension not implemented')
            return None

    def clearFolder(self, pathFolder):
        if not os.path.exists(pathFolder):
            os.makedirs(pathFolder)
        for the_file in os.listdir(pathFolder):
            file_path = os.path.join(pathFolder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                else:
                    shutil.rmtree(pathFolder)
            except Exception as e:
                print(e+'\n')
        
    def __init__(self, fullPath, clear = False):

        # if os.path.exists(fullPath): # Already exists
        if fullPath == '': # Placeholder
            self.name = ''
            self.nameNoExtension = ''
            self.path = ''
            self.fullPath = ''
            self.isDir = False
        elif os.path.isfile(fullPath):  # If is a file
            self.isDir = False
            self.name = fullPath.split('/')[-1]
            self.nameNoExtension = '.'.join((self.name).split('.')[:-1])
            self.extension = (self.name).split('.')[-1]
            self.path = '/'.join(fullPath.split('/')[0:-1])+'/'
            self.fullPath = fullPath
            self.type = self.findType()
        else:  # Is a folder
            self.type = 'dir'
            self.isDir = True

            # Remove potential end slash
            if fullPath[-1] == '/':
                fullPath = fullPath[:-1]

            self.name = fullPath.split('/')[-1]
            self.nameNoExtension = self.name
            self.path = '/'.join(fullPath.split('/')[0:-1])+'/'
            self.fullPath = fullPath+'/'
            
            if clear:
                self.clearFolder(fullPath)

            if os.path.exists(fullPath): # Already exists
                self.updateFiles()
            else:
                self.fileNames = []
                self.nbFiles = 0
                self.files = []

    def updateFiles(self):
        if self.isDir:
            self.fileNames=[f for f in os.listdir(self.fullPath) if f[-4:] in ['.tif', 'tiff', '.viv']]
            self.fileNames.sort()

            self.nbFiles = len(self.fileNames)
            self.files = []

            for fileName in self.fileNames:
                if os.path.isfile(self.fullPath+fileName): # If is a file
                    self.files.append(File(self.fullPath+fileName))
                else:
                    print('Not a file, but a folder')

    def read(self, ind = 0):
        if self.isDir:
            if isinstance(ind, (float, int)):
                return self.files[ind].read()
            else:
                img = self.read(0)
                stack = np.zeros([len(ind), img.shape[0], img.shape[1]], dtype = 'float32')
                for (i,j) in enumerate(ind):
                    stack[i,:,:] = self.files[j].read()
                return stack

        else:
            if self.type == 'tif':
                return tifffile.imread(self.fullPath)
            elif self.type == 'viv':
                with open(self.fullPath,'r') as img:
                    img.seek(2048) # Skip header
                    data=np.fromfile(img, dtype=np.uint16)
                    if data.shape[0] == 2048 * 1536:
                        return data.reshape(1536, 2048)
                    elif data.shape[0] == 1024 * 768:
                        return data.reshape(768, 1024)
                    else:
                        print('Viv file size not implemented')
                        return None
            else:
                print('File extension not implemented')
                return None

    def getDimensions(self):
        try:
            img = self.read()
            return img.shape
        except:
            print('File extension not implemented')
            return None


    def readAll(self):
        if self.isDir:
            img = self.read(0)
            stack = np.zeros([self.nbFiles, img.shape[0], img.shape[1]], dtype = 'float32')
            for i in range(self.nbFiles):
                stack[i,:,:] = self.files[i].read()
            return stack
        else:
            print('Can''t read all, File is a folder')
            return None

    def create(self):
        if self.isDir and (not os.path.exists(self.fullPath)):
            os.makedirs(self.fullPath)

    def saveTiff(self, img, ind=0, suffix = '', exact_prefix = '', type = 'float32'):
        self.create()
        if suffix != '':
            suffix = '_'+suffix
        if self.isDir:
            if exact_prefix == '':
                tifffile.imsave(self.fullPath+self.name+'_'+str(ind).zfill(4)+suffix+'.tif', img.astype(type))
            else:
                tifffile.imsave(self.fullPath+exact_prefix+'_'+str(ind).zfill(4)+suffix+'.tif', img.astype(type))
            self.updateFiles()

    def saveTiffStack(self, vol, suffix = '', ind=[], ind_offset = 0, type = 'float32'):
        self.create()
        if self.isDir:
            if len(ind) == 0:
                for i in range(vol.shape[0]):
                    self.saveTiff(vol[i,:,:], ind = i+ind_offset, suffix = suffix, type = type)
            else:
                for i in range(vol.shape[0]):
                    self.saveTiff(vol[i,:,:], ind = ind[i]+ind_offset, suffix = suffix, type = type)

# TESTS DOWN THERE

if __name__ == '__main__':


	test = File('/run/media/eboigne/Data1_EB/med_school_CnF_andAfter/XCT_BOS_CTC/Ab_PMB_usedToGetCORforStanfordS/proj02_flat01_averagedTR_XstepAngles/')

	print(test.read(1))
	print(test.files[1].fullPath)

	folderOut = File('/home/eboigne/Desktop/test/')
	folderOut.create()

	img = test.read(1,)
	folderOut.saveTiff(img,10,'slice1')
