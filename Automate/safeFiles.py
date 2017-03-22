
def save_files(test, files,obstacles, Resolutionx, Resolutiony, numberObj, FOVx,FOVy):
    #test= 'mytestfile.hdf5'
    #test = 'test'+time.strftime('%F-%T')+'.hdf5'
    exist = False

    for s in files:
        if test == s:
            exist = True
            print('exists')
            #f.close()
            f = h5py.File(test)
            if 'obstacles' in f:
                r= obstacles.shape
                print(f['obstacles'].shape)
                f['obstacles'].resize(r)
                f['obstacles'][...] = obstacles
                data= f['obstacles']
                print(f['obstacles'][:])
                #groupName = ''

            else:    
                data = f.create_dataset('obstacles', data =obstacles, dtype = 'f', chunks=True )


    if exist == False:
        f =h5py.File(test, 'w')
        data = f.create_dataset('obstacles', data=obstacles,dtype='f', chunks = True)

        
    data.attrs['Resolution']= [Resolutionx, Resolutiony]
    data.attrs['NumberObjects']=numberObj
    data.attrs['FOV']=[FOVx,FOVy]    
    f.close()