from deepface import DeepFace


models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

#face verification
result = DeepFace.verify(
  img1_path = "img1.jpg",
  img2_path = "img2.jpg",
  model_name = models[0],
)

#face recognition
dfs = DeepFace.find(
  img_path = "img1.jpg",
  db_path = "C:/workspace/my_db", 
  model_name = models[1],
)

#embeddings
embedding_objs = DeepFace.represent(
  img_path = "img.jpg",
  model_name = models[2],
)