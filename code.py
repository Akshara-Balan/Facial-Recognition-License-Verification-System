import cv2
import face_recognition
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from datetime import datetime

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

# List of known image paths
image_paths = [.........""# List of known image paths here"".....]

# Loop through image paths to load images and encodings
for image_path in image_paths:
try:
known_person_image = face_recognition.load_image_file(image_path)
known_person_encoding = face_recognition.face_encodings(known_person_image)[0]
known_face_encodings.append(known_person_encoding)

# Extract name from image path
name = image_path.split("/")[-1].split(".")[0]
known_face_names.append(name)
except Exception as e:
print(f"Error loading encoding for {image_path}: {e}")

# OpenCV function to choose file from disk
def get_image_path():
return input("Enter the file path or URL of the image you want to check: ")
def display_image(image):
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Now, you can use the trained model to predict names based on license numbers
def predict_details(license_number):
license_encoded = encoder.transform([license_number])
license_encoded_scaled = scaler.transform(license_encoded.reshape(1, -1))
predicted_name = classifier.predict(license_encoded_scaled)
index = df.index[df['FULL_NAME'] == predicted_name[0]].tolist()[0]
predicted_gender = gender[index]
predicted_dob = dob[index]
predicted_issue_date = issue_date[index]
predicted_validity=validity[index]
return predicted_name[0], predicted_gender, predicted_dob, predicted_issue_date,predicted_validity

# Call the modified function
def compare_faces_and_predict(image_path):
if not verify_image(image_path):
return
frame = cv2.imread(image_path)
face_locations = face_recognition.face_locations(frame)
face_encodings = face_recognition.face_encodings(frame, face_locations)
current_date = datetime.now()
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
name = "Unknown"
if True in matches:
first_match_index = matches.index(True)
name = known_face_names[first_match_index]
cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
if name == "Unknown":
print("This person has no license.")
else:
predicted_name, predicted_gender, predicted_dob, predicted_issue_date, predicted_validity =
predict_details(name)
print("Predicted name:", predicted_name)
print("Predicted gender:", predicted_gender)
print("Predicted date of birth:", predicted_dob)
print("Predicted issue date:", predicted_issue_date)
print("License validity:", predicted_validity)

# Check if the license has expired
validity_date = datetime.strptime(predicted_validity, "%d/%m/%Y")
if current_date > validity_date:
print("License has expired.")
else:
print("License is valid until", predicted_validity)
display_image(frame)
def verify_image(image_path):
frame = cv2.imread(image_path)
if frame is None:
print("Error: Unable to read the image. Please check the file path.")
return False
display_image(frame)
verification = input("Is the displayed image the same as the one you wanted to check? (yes/no): ")
if verification.lower() == "yes":
print("Verification successful.")
return True
else:
print("Verification failed.")
return False

#read dataset
df = pd.read_csv('mockdataset.csv')

#preprocess data
encoder=LabelEncoder()
df['LicenseEncoded']=encoder.fit_transform(df['DL_NO'])
X=df[['LicenseEncoded']]
y=df['FULL_NAME']
gender=df['GENDER']
dob=df['DATE_OF_BIRTH']
issue_date=df['ISSUE_DATE']
validity=df['VALIDITY_DATE']

#standardize features
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#train naive bayes
classifier=GaussianNB()
classifier.fit(X_scaled,y)

# Call the function with the image path obtained from user input
image_path = get_image_path()
compare_faces_and_predict(image_path)
