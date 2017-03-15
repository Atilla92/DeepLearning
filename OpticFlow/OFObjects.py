

def OFObject (fov_x, fov_y, res_x, res_y):
phis   = np.linspace(-fov_x, fov_x, res_x)
thetas = np.linspace(-fov_y, fov_y, res_y)
layout = np.array([[phi, theta] 
                   for theta in thetas
                   for phi in phis])