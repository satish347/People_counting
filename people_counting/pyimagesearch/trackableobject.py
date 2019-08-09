import datetime
class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		#print(self.objectID)

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False
		self.enter = False
		self.exit = False
		#self.enter1 = False
		#self.exit1 = False
		#self.enter2 = False
		#self.exit2 = False
		#self.enter3 = False
		#self.exit3 = False
		#self.enter4 = False
		#self.exit4 = False
