"""
Apply Color Selection
Apply Canny edge detection.
Apply gray scaling to the images.
Apply Gaussian smoothing.
Perform Canny edge detection.
Determine the region of interest.
Apply Hough transform.
Average and extrapolating the lane lines.
Apply on video streams.
"""
import cv2
import numpy as np

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

def color_selection(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    mask = cv2.bitwise_or(mask_white, mask_yellow)
    
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)     
    return blurred

def apply_canny_edge_detection(blurred):
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    
    polygon = np.array([[
        (0, height),
        (width / 2 - 50, height / 2 + 50),
        (width / 2 + 50, height / 2 + 50),
        (width, height),
    ]], dtype=np.int32)
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def hough_transform(image):
    _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    
    lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=200)
    return lines

def average_and_extrapolate(image, lines):
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else 0
        intercept = y1 - slope * x1
        
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    
    left_line = np.mean(left_lines, axis=0) if len(left_lines) > 0 else None
    right_line = np.mean(right_lines, axis=0) if len(right_lines) > 0 else None
    
    height = image.shape[0]
    y1 = height
    y2 = int(height / 2)
    
    def extrapolate_line(slope, intercept):
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return (x1, y1, x2, y2)
    
    left_line_points = extrapolate_line(left_line[0], left_line[1]) if left_line is not None else None
    right_line_points = extrapolate_line(right_line[0], right_line[1]) if right_line is not None else None
    
    return left_line_points, right_line_points

def draw_lines(image, left_line, right_line):
    if left_line is not None:
        cv2.line(image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 10)  # Red line for left lane
    if right_line is not None:
        cv2.line(image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 10)  # Blue line for right lane
    return image

def process_video(video_path):
    cap = load_video(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Step 1: Apply color selection to filter out the lanes (white and yellow)
        filtered_image = color_selection(frame)
        
        # Step 2: Preprocess image (convert to grayscale + blur)
        blurred_image = preprocess_image(filtered_image)
        
        # Step 3: Apply Canny edge detection
        edges = apply_canny_edge_detection(blurred_image)
        
        # Step 4: Mask region of interest (focus on the lane area)
        roi_image = region_of_interest(edges)
        
        # Step 5: Detect lines using Hough Transform
        lines = hough_transform(roi_image)
        
        # Step 6: Average and extrapolate the lane lines
        if lines is not None:
            left_line, right_line = average_and_extrapolate(frame, lines)
        else:
            left_line, right_line = None, None
        
        # Step 7: Draw the lane lines on the image
        final_image = draw_lines(frame, left_line, right_line)
        
        # Display the result
        cv2.imshow("Lane Detection", final_image)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'vid1.mp4'  # Replace this with the path to your video file
    process_video(video_path)
