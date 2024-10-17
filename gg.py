import pandas as pd
from PIL import Image
import datetime
import sys

# Load the CSV file containing rainfall data
rain_data = pd.read_csv('static/trends.csv')

# Function to map rainfall to a gradient of red
def rainfall_to_gradient_red(rainfall):
    if 1 <= rainfall <= 3:
        return (255, 195, 0)   # #FF6107 for rainfall between 1 to 3
    elif 4 <= rainfall <= 8:
        return (255, 87, 51)  # #E9290F for rainfall between 4 to 8
    elif 9 <= rainfall <= 13:
        return (255, 0, 0)   # #C40018 for rainfall between 9 to 15
    elif rainfall >= 14:
        return (128, 0, 0)    # #840404 for rainfall 16 and above
    else:
        return (218, 247, 166 )    # Default for undefined cases



# Function to replace red pixels in the image with a specified red gradient color
def replace_red_with_gradient_red(image, red_color):
    img = image.convert("RGBA")
    data = img.getdata()

    new_data = []
    for item in data:
        # Detect if the pixel is predominantly red
        if item[0] > 150 and item[1] < 100 and item[2] < 100:
            new_data.append(red_color + (item[3],))  # Replace red with the new gradient shade
        else:
            new_data.append(item)

    img.putdata(new_data)
    return img

# Function to process images for a specific date
def process_images_for_date(input_date):
    date_obj = datetime.datetime.strptime(input_date, '%Y-%m-%d')
    filtered_data = rain_data[rain_data['日付'] == input_date]

    for index, row in filtered_data.iterrows():
        region = row['場所']
        rainfall = row['評価']
        
        try:
            img = Image.open(f'旭区地図/10/{region}.png')  # Path to the image file
            red_color = rainfall_to_gradient_red(rainfall)  # Get the corresponding red color
            modified_img = replace_red_with_gradient_red(img, red_color)  # Modify the image
            modified_img.save(f'static/images/{region}.png')  # Save the modified image
            print(f"Image for {region} has been customized and saved.")
            
        except FileNotFoundError:
            print(f"Image for {region} not found.")
        except Exception as e:
            print(f"Error processing {region}: {e}")

# Function to run the image processing
def run_processing(input_date):
    process_images_for_date(input_date)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_date = str(sys.argv[1])  # Get the date from command line argument
        process_images_for_date(input_date)  # Process images for the specified date
        print(input_date)
    else:
        print("No date provided.")
