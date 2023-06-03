import cv2
import numpy as np

def count_people(image):
  """Counts the number of people in an image.

  Args:
    image: The input image.

  Returns:
    The number of people in the image.
  """

  # Convert the image to grayscale.
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply a Gaussian blur to the image to smooth it out.
  blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

  # Threshold the image to create a binary image.
  thresholded_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)[1]

  # Find the contours in the binary image.
  contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Initialize the count of people to 0.
  people_count = 0

  # Loop over the contours.
  for contour in contours:
    # Check if the contour is large enough to be a person.
    if cv2.contourArea(contour) > 1000:
      # Increment the count of people.
      people_count += 1

  # Return the number of people.
  return people_count

if __name__ == "__main__":
  # Load the image.
  image = cv2.imread("feed.png")

  # Count the number of people in the image.
  people_count = count_people(image)

  # Display the number of people.
  cv2.putText(image, str(people_count), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

  # Show the image.
  cv2.imshow("People Counter", image)
  cv2.waitKey(0)
