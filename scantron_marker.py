import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from datetime import datetime

class ScantronMarker:
    def __init__(self):
        # Configuration parameters
        self.num_questions = 30  # Number of questions
        self.options_per_question = 5  # Options per question (A-E)
        self.answer_key = {}  # Will store the correct answers
        self.results = []
        
    def set_answer_key(self, answers):
        """Set the answer key"""
        self.answer_key = answers
        print(f"Answer key set with {len(answers)} questions")
        
    def process_image(self, image_path):
        """Process a scantron sheet image"""
        print(f"Processing image: {image_path}")
        
        # Load the image
        if not os.path.exists(image_path):
            print(f"Error: Image file not found - {image_path}")
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load image - {image_path}")
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find bubbles (based on area and circularity)
        bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50 and area < 300:  # Filter by area (adjust as needed)
                x, y, w, h = cv2.boundingRect(contour)
                if 0.8 < w/h < 1.2:  # Approximately square/circular
                    center = (x + w//2, y + h//2)
                    radius = (w + h) // 4
                    bubbles.append((center, radius, area))
        
        # Detect which bubbles are marked
        marked_bubbles = []
        for (center, radius, area) in bubbles:
            x, y = center
            # Check the intensity in the bubble area
            mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            mean_intensity = cv2.mean(binary, mask=mask)[0]
            
            # If the mean intensity is high enough, the bubble is marked
            if mean_intensity > 200:
                marked_bubbles.append((x, y))
        
        # Map marked bubbles to answers
        student_answers = self.map_bubbles_to_answers(marked_bubbles)
        
        # Grade the sheet
        result = self.grade_sheet(student_answers)
        
        # Visualize the results
        self.visualize_results(image, student_answers, result)
        
        # Add to results for summary report
        self.results.append(result)
        
        return result
        
    def map_bubbles_to_answers(self, marked_bubbles):
        """Map marked bubbles to question answers"""
        # This is a simplified approach - in a production system,
        # this would need calibration based on the specific sheet layout
        
        # Sort bubbles by y-coordinate (roughly corresponding to question rows)
        sorted_bubbles = sorted(marked_bubbles, key=lambda b: b[1])
        
        # Group bubbles into rows (questions)
        rows = []
        current_row = []
        last_y = None
        
        for x, y in sorted_bubbles:
            if last_y is None or abs(y - last_y) < 20:  # Same row threshold
                current_row.append((x, y))
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [(x, y)]
            last_y = y
            
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))
        
        # Map to answers (A, B, C, D, E)
        student_answers = {}
        options = ['A', 'B', 'C', 'D', 'E']
        
        for i, row in enumerate(rows):
            if i < self.num_questions:
                # Determine which option was selected in this row
                if row:  # At least one bubble marked
                    # Take the leftmost marked bubble as the answer
                    # (this assumes options are arranged left to right: A, B, C, D, E)
                    leftmost_x = min(x for x, _ in row)
                    position = 0
                    for x, _ in row:
                        if x == leftmost_x:
                            break
                        position += 1
                    
                    if position < len(options):
                        q_num = i + 1
                        student_answers[q_num] = options[position]
        
        return student_answers
    
    def grade_sheet(self, student_answers):
        """Grade the sheet by comparing to answer key"""
        if not self.answer_key:
            print("Error: Answer key not set")
            return None
            
        correct_count = 0
        incorrect_count = 0
        unanswered_count = 0
        question_results = {}
        
        for q_num, correct_answer in self.answer_key.items():
            if q_num in student_answers:
                student_answer = student_answers[q_num]
                if student_answer == correct_answer:
                    correct_count += 1
                    question_results[q_num] = "Correct"
                else:
                    incorrect_count += 1
                    question_results[q_num] = "Incorrect"
            else:
                unanswered_count += 1
                question_results[q_num] = "Unanswered"
        
        total_questions = len(self.answer_key)
        score = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        
        result = {
            "Total Questions": total_questions,
            "Correct": correct_count,
            "Incorrect": incorrect_count,
            "Unanswered": unanswered_count,
            "Score": score,
            "Question Results": question_results,
            "Student Answers": student_answers
        }
        
        return result
    
    def visualize_results(self, image, student_answers, result):
        """Visualize the grading results on the image"""
        # Create a copy to draw on
        marked_image = image.copy()
        
        # Add a sidebar for results
        h, w = marked_image.shape[:2]
        sidebar_width = 300
        sidebar = np.ones((h, sidebar_width, 3), dtype=np.uint8) * 255
        
        # Add score and results to sidebar
        cv2.putText(sidebar, f"Score: {result['Score']:.1f}%", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(sidebar, f"Correct: {result['Correct']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
        cv2.putText(sidebar, f"Incorrect: {result['Incorrect']}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(sidebar, f"Unanswered: {result['Unanswered']}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 0), 2)
        
        # Add student answers to sidebar
        cv2.putText(sidebar, "Student Answers:", 
                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Show first 20 answers in the sidebar
        y_pos = 190
        for i, (q_num, answer) in enumerate(sorted(student_answers.items())[:20]):
            color = (0, 128, 0) if result["Question Results"][q_num] == "Correct" else (0, 0, 255)
            cv2.putText(sidebar, f"Q{q_num}: {answer}", 
                       (10, y_pos + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Combine image with sidebar
        output_image = np.hstack((marked_image, sidebar))
        
        # Save the visualization
        output_name = f"graded_sheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(output_name, output_image)
        print(f"Visualization saved as {output_name}")
        
        # Display result
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Graded Scantron Sheet')
        plt.show()
        
        return output_image
    
    def generate_summary_report(self):
        """Generate a summary report of student performance"""
        if not self.results:
            print("No results to report")
            return
            
        num_sheets = len(self.results)
        print(f"\n=== SUMMARY REPORT ===")
        print(f"Total sheets processed: {num_sheets}")
        
        # Overall performance statistics
        avg_score = sum(r["Score"] for r in self.results) / num_sheets
        print(f"Average score: {avg_score:.2f}%")
        
        # Question-by-question performance
        print("\nQuestion Performance:")
        
        # Collect question stats
        q_stats = {}
        for result in self.results:
            for q_num, status in result["Question Results"].items():
                if q_num not in q_stats:
                    q_stats[q_num] = {"Correct": 0, "Incorrect": 0, "Unanswered": 0}
                q_stats[q_num][status] += 1
        
        # Convert to percentages
        for q_num, stats in q_stats.items():
            correct_pct = (stats["Correct"] / num_sheets) * 100
            incorrect_pct = (stats["Incorrect"] / num_sheets) * 100
            unanswered_pct = (stats["Unanswered"] / num_sheets) * 100
            print(f"Q{q_num}: Correct {correct_pct:.1f}%, Incorrect {incorrect_pct:.1f}%, Unanswered {unanswered_pct:.1f}%")
        
        # Create a visualization of the results
        q_nums = sorted(q_stats.keys())
        correct_pcts = [(q_stats[q]["Correct"] / num_sheets) * 100 for q in q_nums]
        
        plt.figure(figsize=(12, 6))
        plt.bar(q_nums, correct_pcts, color='green')
        plt.xlabel('Question Number')
        plt.ylabel('Percentage Correct (%)')
        plt.title('Question Performance Summary')
        plt.xticks(q_nums)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the report
        report_name = f"question_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        plt.savefig(report_name)
        plt.close()
        print(f"\nSummary report saved as {report_name}")
        
        # Create a more detailed report in CSV format
        df = pd.DataFrame([{
            "Question": q,
            "Correct %": (q_stats[q]["Correct"] / num_sheets) * 100,
            "Incorrect %": (q_stats[q]["Incorrect"] / num_sheets) * 100,
            "Unanswered %": (q_stats[q]["Unanswered"] / num_sheets) * 100
        } for q in q_nums])
        
        csv_name = f"question_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Detailed report saved as {csv_name}")


# Function to demonstrate usage
def main():
    # Create a ScantronMarker instance
    marker = ScantronMarker()
    
    # Generate a random answer key for 30 questions
    options = ['A', 'B', 'C', 'D', 'E']
    answer_key = {i: random.choice(options) for i in range(1, 31)}
    
    print("Randomly generated answer key:")
    for q_num, answer in answer_key.items():
        print(f"Question {q_num}: {answer}")
    
    # Set the answer key
    marker.set_answer_key(answer_key)
    
    # Define a list of image files to process (replace with your actual image files)
    # These should be the images of scantron sheets that you've saved manually
    image_files = []
    
    # Check the current directory for image files
    for file in os.listdir('.'):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            image_files.append(file)
    
    if not image_files:
        print("\nNo image files found in the current directory.")
        print("Please convert your PDF to images manually and save them in this directory.")
        print("Supported formats: PNG, JPG, JPEG, TIF, TIFF, BMP")
        print("\nHow to convert PDF to images:")
        print("1. Open the PDF in a viewer like Adobe Reader or browser")
        print("2. Take screenshots or use Print > Save as Image")
        print("3. Save the images in this directory")
        print("4. Run this script again")
        return
    
    print(f"\nFound {len(image_files)} image files to process: {', '.join(image_files)}")
    
    # Process each image file
    for image_file in image_files:
        marker.process_image(image_file)
    
    # Generate summary report
    marker.generate_summary_report()

if __name__ == "__main__":
    main()
