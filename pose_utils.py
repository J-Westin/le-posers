import numpy as np
import json
import os
from PIL import Image
import keras

from matplotlib import pyplot as plt

# deep learning framework imports
try:
	from keras.utils import Sequence
	from keras.preprocessing import image as keras_image
	has_tf = True
except ModuleNotFoundError:
	has_tf = False

try:
	import torch
	from torch.utils.data import Dataset
	from torchvision import transforms
	has_pytorch = True
except ImportError:
	has_pytorch = False


class Camera:

	"""" Utility class for accessing camera parameters. """

	fx = 0.0176  # focal length[m]
	fy = 0.0176  # focal length[m]
	nu = 1920  # number of horizontal[pixels]
	nv = 1200  # number of vertical[pixels]
	ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
	ppy = ppx  # vertical pixel pitch[m / pixel]
	fpx = fx / ppx  # horizontal focal length[pixels]
	fpy = fy / ppy  # vertical focal length[pixels]
	k = [[fpx,   0, nu / 2],
		 [0,   fpy, nv / 2],
		 [0,	 0,	  1]]
	K = np.array(k)


def process_json_dataset(root_dir):
	with open(os.path.join(root_dir, 'train.json'), 'r') as f:
		train_images_labels = json.load(f)

	with open(os.path.join(root_dir, 'test.json'), 'r') as f:
		test_image_list = json.load(f)

	with open(os.path.join(root_dir, 'real_test.json'), 'r') as f:
		real_test_image_list = json.load(f)

	partitions = {'test': [], 'train': [], 'real_test': []}
	labels = {}

	for image_ann in train_images_labels:
		partitions['train'].append(image_ann['filename'])
		labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango'], 'r': image_ann['r_Vo2To_vbs_true']}

	for image in test_image_list:
		partitions['test'].append(image['filename'])

	for image in real_test_image_list:
		partitions['real_test'].append(image['filename'])

	return partitions, labels

def quat2dcm(q):

	""" Computing direction cosine matrix from quaternion, adapted from PyNav. """

	# normalizing quaternion
	q = q/np.linalg.norm(q)

	q0 = q[0]
	q1 = q[1]
	q2 = q[2]
	q3 = q[3]

	dcm = np.zeros((3, 3))

	dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
	dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
	dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

	dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
	dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

	dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
	dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

	dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
	dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

	return dcm


def project(q, r):

	""" Projecting points to image frame to draw axes """

	# reference points in satellite frame for drawing axes
	p_axes = np.array([[0, 0, 0, 1],
					   [1, 0, 0, 1],
					   [0, 1, 0, 1],
					   [0, 0, 1, 1]])
	points_body = np.transpose(p_axes)

	# transformation to camera frame
	pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
	p_cam = np.dot(pose_mat, points_body)

	# getting homogeneous coordinates
	points_camera_frame = p_cam / p_cam[2]

	# projection to image plane
	points_image_plane = Camera.K.dot(points_camera_frame)

	x, y = (points_image_plane[0], points_image_plane[1])
	return x, y

def project_sc2im(r_sc, q, r):

	""" Projects r_sc = (x_sc, y_sc, z_sc) from the spacecraft frame to the image frame """
	
	x_sc, y_sc, z_sc = r_sc
	r_sc_hmg = np.array([[x_sc],[y_sc],[z_sc],[1.]])
	
	proj_pose = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
	
	r_cam = np.dot(proj_pose, r_sc_hmg)
	r_cam_hmg = r_cam / r_cam[2]
	
	p_im = np.dot(Camera.K, r_cam_hmg)
	x_im, y_im, z_im = list(p_im)
	
	return x_im, y_im

def corner_points_im(q, r, margins):
	X, Y = [], []
	mrg_x, mrg_y, mrg_z = margins
	
	for x in [-0.4-mrg_x, 0.4+mrg_x]:
		for y in [-0.4-mrg_y, 0.4+mrg_y]:
			for z in [0.-mrg_z, 0.31+mrg_z]:
				x_im, y_im = project_sc2im([x, y, z], q, r)
				X.append(x_im)
				Y.append(y_im)
	
	return X, Y

class SatellitePoseEstimationDataset:

	""" Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

	def __init__(self, root_dir='/datasets/speed_debug'):
		self.partitions, self.labels = process_json_dataset(root_dir)
		self.root_dir = root_dir

	def get_image(self, i=0, split='train'):

		""" Loading image as PIL image. """

		img_name = self.partitions[split][i]
		img_name = os.path.join(self.root_dir, 'images', split, img_name)
		image = Image.open(img_name).convert('RGB')
		return image

	def get_pose(self, i=0):

		""" Getting pose label for image. """

		img_id = self.partitions['train'][i]
		q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
		return q, r

	def visualize(self, i, partition='train', ax=None):

		""" Visualizing image, with ground truth pose with axes projected to training image. """

		if ax is None:
			ax = plt.gca()
		img = self.get_image(i)
		ax.imshow(img)

		# no pose label for test
		if partition == 'train':
			q, r = self.get_pose(i)
			xa, ya = project(q, r)
			ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
			ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
			ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')

		return
	
	def target_bbox_coordinates(self, i, margins):
		
		""" return target bounding box pixel coordinates as four floats: x_min, x_max, y_min, y_max. """
		
		q, r = self.get_pose(i)
		x_crn, y_crn = corner_points_im(q, r, margins)
		
		x_min, x_max = min(x_crn), max(x_crn)
		y_min, y_max = min(y_crn), max(y_crn)
		x_min, x_max, y_min, y_max = x_min[0], x_max[0], y_min[0], y_max[0]
		
		x_min = max(x_min, 0)
		x_max = min(x_max, Camera.nu)
		y_min = max(y_min, 0)
		y_max = min(y_max, Camera.nv)
		
		return [x_min/Camera.nu, x_max/Camera.nu, y_min/Camera.nv, y_max/Camera.nv]
	
	def generate_bbox_json(self, margins, output_filename="train_bbox.json"):
		bbox_data_list = []
		
		for k in range(len(self.labels)):
			current_filename = self.partitions["train"][k]
			current_bbox_label = self.target_bbox_coordinates(k, margins)
			current_json_entry = {"filename": current_filename, "bbox":current_bbox_label}
			
			bbox_data_list.append(current_json_entry)
		
		filepath = os.path.join(self.root_dir, output_filename)
		
		output_file = open(filepath, "w")
		json.dump(bbox_data_list, output_file)
		output_file.close()
		
		return


if has_pytorch:
	class PyTorchSatellitePoseEstimationDataset(Dataset):

		""" SPEED dataset that can be used with DataLoader for PyTorch training. """

		def __init__(self, split='train', speed_root='', transform=None):

			if not has_pytorch:
				raise ImportError('Pytorch was not imported successfully!')

			if split not in {'train', 'test', 'real_test'}:
				raise ValueError('Invalid split, has to be either \'train\', \'test\' or \'real_test\'')

			with open(os.path.join(speed_root, split + '.json'), 'r') as f:
				label_list = json.load(f)

			self.sample_ids = [label['filename'] for label in label_list]
			self.train = split == 'train'

			if self.train:
				self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
							   for label in label_list}
			self.image_root = os.path.join(speed_root, 'images', split)

			self.transform = transform

		def __len__(self):
			return len(self.sample_ids)

		def __getitem__(self, idx):
			sample_id = self.sample_ids[idx]
			img_name = os.path.join(self.image_root, sample_id)

			# note: despite grayscale images, we are converting to 3 channels here,
			# since most pre-trained networks expect 3 channel input
			pil_image = Image.open(img_name).convert('RGB')

			if self.train:
				q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
				y = np.concatenate([q, r])
			else:
				y = sample_id

			if self.transform is not None:
				torch_image = self.transform(pil_image)
			else:
				torch_image = pil_image

			return torch_image, y
else:
	class PyTorchSatellitePoseEstimationDataset:
		def __init__(self, *args, **kwargs):
			raise ImportError('Pytorch is not available!')

if has_tf:
	class KerasDataGenerator(Sequence):

		""" DataGenerator for Keras to be used with fit_generator (https://keras.io/models/sequential/#fit_generator)"""

		def __init__(self, preprocessor, label_list, speed_root, 
					batch_size=32, dim=(224, 224), n_channels=3, 
					shuffle=True, randomRotations = False, seed = 1,
					crop = False, cropper_model = None):

			# loading dataset
			self.image_root = os.path.join(speed_root, 'images', 'train')

			# Initialization
			self.preprocessor = preprocessor
			self.dim = dim
			self.batch_size = batch_size
			self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
										 for label in label_list}
			self.list_IDs = [label['filename'] for label in label_list]
			self.n_channels = n_channels
			self.shuffle = shuffle
			self.indexes = None
			self.randomRotations = randomRotations
			self.seed = seed
			self.crop = crop
			self.cropper_model = cropper_model
			self.on_epoch_end()

		def __len__(self):

			""" Denotes the number of batches per epoch. """

			return int(np.floor(len(self.list_IDs) / self.batch_size))

		def __getitem__(self, index):

			""" Generate one batch of data """

			# Generate indexes of the batch
			indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

			# Find list of IDs
			list_IDs_temp = [self.list_IDs[k] for k in indexes]

			# Generate data
			X, y = self.__data_generation(list_IDs_temp)

			return X, y

		def on_epoch_end(self):

			""" Updates indexes after each epoch """

			self.indexes = np.arange(len(self.list_IDs))
			if self.shuffle:
				np.random.shuffle(self.indexes)

		def random_image_rotation(self, img, q, r):
			# generate a random angle
			angle = np.random.uniform(low = 0.0, high = 360.0)
			
			# rotate image
			img = img.rotate(-angle)
			
			# compute new location
			angle = angle/180*np.pi
			r = [np.cos(angle)*r[0] - np.sin(angle)*r[1],
				 np.sin(angle)*r[0] + np.cos(angle)*r[1],
				 r[2]]
			
			# compute new orientation
			rotationAxis = [0, 0, 1]
			qRot = [np.cos(angle/2),
					np.sin(angle/2)*rotationAxis[0],
					np.sin(angle/2)*rotationAxis[1],
					np.sin(angle/2)*rotationAxis[2]]
			w0, x0, y0, z0 = q
			w1, x1, y1, z1 = qRot
			q = [-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
				  x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
				 -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
				  x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0]
					  
			# saves a bunch of images for debugging
			if False:
				prec = 3
				# fig, axes = plt.figure(figsize=(10, 5))
				plt.figure('test')
				plt.title('Index: '+ str(ID) +
							 '\nRotation angle: ' + str((round(angle*180/np.pi))) + ' degree' +
							 '\nPosition: [' + str(round(r[0], prec)) + ', ' + str(round(r[1], prec))+', '+str(round(r[2], prec))+']'+
							 '\nOrientation: [' + str(round(q[0], prec)) + ', ' + str(round(q[1], prec))+', '+str(round(q[2], prec))+', '+str(round(q[3], prec))+']')
				ax = plt.gca()
				ax.imshow(img)
				xa, ya = project(q, r)
				ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=6, color='r')
				ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=6, color='g')
				ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=6, color='b')
				ax.axis('off')
				plt.savefig(f'TestOutput_{ID}.png', dpi = 1000, bbox_inches = 'tight')
				plt.close() 

			return img, q, r

		def crop_image(self, img, bbox_imgshape = (240, 150)):
			"""
			Crops image in-place and returns coordinates of cropping 
			bounding box which is an input for the neural network
			"""
			img_height, img_width = np.array(img, dtype=float)[:,:].shape
			
			# Preprocess image for cropper and obtain corners
			current_image_pil = img.resize(bbox_imgshape, resample=Image.BICUBIC)
			current_image_arr = np.array(current_image_pil, dtype=float)/256.
			
			coordinates = self.cropper_model.predict(np.expand_dims(np.expand_dims(current_image_arr[:,:],axis=2),axis=0))

#					plt.ioff()
			if False:
				plt.imsave(f'TestOutput_{ID}_O.png',img, dpi = 1000)

			
			left=int(np.minimum(coordinates[0,0],coordinates[0,1])*img_width)
			right=int(np.maximum(coordinates[0,0],coordinates[0,1])*img_width)
			lower=int(np.minimum(coordinates[0,2],coordinates[0,3])*img_height)
			upper=int(np.maximum(coordinates[0,2],coordinates[0,3])*img_height)

			len_dif=abs(left-right)-abs(lower-upper)

			if len_dif>0:
				img = img.crop((int(left),int(lower-len_dif/2),int(right),int(upper+len_dif/2)))
			else:
				img = img.crop((int(left+len_dif/2),int(lower),int(right-len_dif/2),int(upper)))

			if False:
				plt.imsave(f'TestOutput_{ID}.png',img, dpi = 1000)

			return coordinates

		def __data_generation(self, list_IDs_temp):

			""" Generates data containing batch_size samples """

			# Initialization
			X = np.empty((self.batch_size, *self.dim, self.n_channels))
			y = np.empty((self.batch_size, 7), dtype=float)
			if self.crop:
				C = np.empty((self.batch_size, 4))

			# Generate data
			for i, ID in enumerate(list_IDs_temp):
				img_path = os.path.join(self.image_root, ID)
				# img = keras_image.load_img(img_path, target_size=self.dim)
#				img = keras_image.load_img(img_path)
				img = Image.open(img_path)
				q, r = self.labels[ID]['q'], self.labels[ID]['r']

				if self.randomRotations:
					img, q, r = self.random_image_rotation(img, q, r)
					
				if self.crop:
					coordinates = self.crop_image(img)

				img = img.resize(self.dim)

				# flatten and output
				x = keras_image.img_to_array(img)
				if self.crop:
					 x = self.preprocessor(np.concatenate([x,x,x], axis=-1, out=None))
				else:
					 x = self.preprocessor(x)
				
				X[i,] = x
				y[i] = np.concatenate([q, r])  
				
				
				if self.crop:
					C[i,] = coordinates
					Xf=[X,C]
				else:
					 Xf=X

			return Xf, y
else:
	class KerasDataGenerator:
		def __init__(self, *args, **kwargs):
			raise ImportError('tensorflow.keras is not available! Please install tensorflow.')


def checkFolders(folders):
	"""
	Check whether folders are present and creates them if necessary
	"""
	for folder in folders:
		if not os.path.exists(folder):
			print(f'Making directory {folder}')
			os.makedirs(folder)

class OutputResults(object):
	def __init__(self, pose_nn):
		self.pose_nn = pose_nn

	def plot_save_losses(self, train_loss, test_loss):
		"""
		Save and plot the losses
		"""
		#save
		np.savetxt(f'{self.pose_nn.output_loc}Losses_v{self.pose_nn.version}.txt', np.array([train_loss, test_loss]), header = 'train_loss test_loss')

		#plot
		plt.plot(np.arange(1, len(train_loss)+1), train_loss, label = 'Train loss')
		plt.plot(np.arange(1, len(test_loss)+1), test_loss, label = 'Test loss')

		#set ylims starting at 0
		ylims = plt.ylim()
		plt.ylim((0, ylims[1]))

		#set y scale to log if extremely large values are observed
		if (np.max(train_loss) - np.min(train_loss)) > 1e2 or (np.max(test_loss) - np.min(test_loss)) > 1e2:
			plt.yscale('log')

		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title(f'Loss progression of version {self.pose_nn.version}')
		plt.legend(loc = 'best')
		plt.grid(alpha = 0.4)

		plt.savefig(f'{self.pose_nn.output_loc}Losses_v{self.pose_nn.version}.png', dpi = 300, bbox_inches = 'tight')
		plt.close()

	def saveLoadModel(self, filename, model=None, save=False, load=False):
		"""
		Load or save a model easily given a path+filename. Filenames must have the format 'model.h5'.
		This saves/load model architecture, weights, loss function, optimizer, and optimizer state.

		Returns:
		model if load=True, else just saves the input model
		"""
		if save:
			print('Saving model to {}'.format(filename))
			model.save(filename)
		if load:
			import keras.losses
			keras.losses.loss_function = self.pose_nn.loss_function
			if not os.path.exists(filename):
				print('Cannot find specified model, check if path or filename is correct')
				return
			print('Loading model from {0}'.format(filename))
			model = keras.models.load_model(filename)
			
			return model