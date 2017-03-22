import h5py
from computations import*
def save_obstacles(test, files,obstacles, Resolutionx, Resolutiony, numberObj, FOVx,FOVy):
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
                f['obstacles'].resize(r)
                f['obstacles'][...] = obstacles
                data= f['obstacles']

            else:    
                data = f.create_dataset('obstacles', data =obstacles, dtype = 'f', chunks=True )


    if exist == False:
        f =h5py.File(test, 'w')
        data = f.create_dataset('obstacles', data=obstacles,dtype='f', chunks = True)

        
    data.attrs['Resolution']= [Resolutionx, Resolutiony]
    data.attrs['NumberObjects']=numberObj
    data.attrs['FOV']=[FOVx,FOVy]    
    f.close()


def rand_samples(maxVel, maxHeading, obstacles, area, SF2, numSamples):
    velocity = [rand_velocity(maxVel) for i in range(numSamples)]
    heading = [rand_heading(maxHeading) for i in range(numSamples)]
    position = [rand_position(obstacles, area, SF2) for i in range(numSamples)]
    samples = np.array([[pos, vel, head] 
                   for pos in position
                   for vel in velocity
                    for head in heading])
    return samples

def save_results(num, test, ofx, ofy, new_heading,position, velocity,heading  ):
    f =h5py.File(test)
    #data = f['obstacles']
    name = 'tryout_'+str(num)
    #print(i[0],i[1],i[2], name)
    if name in f:
        grp = f[name]
    else:
        grp = f.create_group(name)
    if "ofx" in grp:
        r = ofx.shape
        grp['ofx'].resize(r)
        grp['ofx'][...]=ofx
    else:
        grp.create_dataset("ofx",data= ofx,dtype='f', chunks = True)
    if "ofy" in grp:
        r = ofy.shape
        grp['ofy'].resize(r)
        grp['ofy'][...]=ofy
    else:
        grp.create_dataset("ofy",data= ofy, dtype='f', chunks = True)
    if 'new_heading' in grp:
        grp['new_heading'][...]=new_heading
    else:
        grp.create_dataset("new_heading", data =new_heading )
    grp.attrs['position']=position
    grp.attrs['velocity']= velocity
    grp.attrs['heading']=heading
    f.close()
    #print(grp['ofx'])
