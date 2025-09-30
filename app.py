
import pandas as pd
import re
import matplotlib.pyplot as plt

# Loading my Excel file
df = pd.read_excel("Record.xlsx")

# Function to classify diagnoses
def classify_diagnosis(diagnosis):
    if pd.isna(diagnosis):
        return 'Unknown'
    
    diag = diagnosis.strip()
    
    # Rule 1: comma logic (after comma > 5 chars → Multiple)
    if ',' in diag:
        after_comma = diag.split(',', 1)[1].strip()
        if len(after_comma) > 5:
            return 'Multiple'
        else:
            return 'Single'
    
    # Rule 2: concatenated diagnoses detected by lowercase->uppercase pattern
    if re.search(r'[a-z][A-Z]', diag):
        return 'Multiple'
    
    # Rule 3: default → Single
    return 'Single'

#  classification
df['Diagnosis_type'] = df['Primary & Provisional'].apply(classify_diagnosis)

# Separate single and multiple diagnoses
single_diag_df = df[df['Diagnosis_type'] == 'Single']
multiple_diag_df = df[df['Diagnosis_type'] == 'Multiple']

# Generate unique lists
unique_single_diagnoses = single_diag_df['Primary & Provisional'].unique()
unique_multiple_diagnoses = multiple_diag_df['Primary & Provisional'].unique()
 
# Save to CSV for review
pd.DataFrame(unique_single_diagnoses, columns=['Single_Diagnoses']).to_csv('single_diagnoses_list.csv', index=False)
pd.DataFrame(unique_multiple_diagnoses, columns=['Multiple_Diagnoses']).to_csv('multiple_diagnoses_list.csv', index=False)

print(f"Total unique single diagnoses: {len(unique_single_diagnoses)}")
print(f"Total unique multiple diagnoses: {len(unique_multiple_diagnoses)}")
print("Lists saved to 'single_diagnoses_list.csv' and 'multiple_diagnoses_list.csv'")


#3.Count of each unique diagnosis in the file NAS with the frequency counts table

# Count frequency of each unique diagnosis
diagnosis_counts = df['Primary & Provisional'].value_counts().reset_index()
diagnosis_counts.columns = ['Diagnosis', 'Count']

# Save to CSV for review
diagnosis_counts.to_csv("diagnosis_frequency_table.csv", index=False)

print(diagnosis_counts.head(10))  # top 10 most common diagnoses

# For Single
single_counts = single_diag_df['Primary & Provisional'].value_counts().reset_index()
single_counts.columns = ['Diagnosis', 'Count']
single_counts.to_csv("single_diagnosis_frequency.csv", index=False)

# For Multiple
multiple_counts = multiple_diag_df['Primary & Provisional'].value_counts().reset_index()
multiple_counts.columns = ['Diagnosis', 'Count']
multiple_counts.to_csv("multiple_diagnosis_frequency.csv", index=False)

import matplotlib.pyplot as plt

# Visualization (bar plot)  Top 20 diagnoses
top20 = diagnosis_counts.head(20)

plt.figure(figsize=(10,6))
plt.barh(top20['Diagnosis'], top20['Count'])
plt.xlabel("Frequency")
plt.ylabel("Diagnosis")
plt.title("Top 20 Diagnoses in NAS")
plt.gca().invert_yaxis()  # so the largest is at the top
plt.tight_layout()
plt.show()

# =========================
# Bar plots for Single vs Multiple Diagnoses
# =========================

# Top 20 Single Diagnoses
top20_single = single_counts.head(20)
plt.figure(figsize=(10,6))
bars = plt.barh(top20_single['Diagnosis'], top20_single['Count'], color='red')
plt.xlabel("Frequency")
plt.ylabel("Single Diagnosis")
plt.title("Top 20 Single Diagnoses")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, str(int(width)), va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Top 20 Multiple Diagnoses
top20_multiple = multiple_counts.head(20)
plt.figure(figsize=(10,6))
bars = plt.barh(top20_multiple['Diagnosis'], top20_multiple['Count'], color='red')
plt.xlabel("Frequency")
plt.ylabel("Multiple Diagnosis")
plt.title("Top 20 Multiple Diagnoses")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, str(int(width)), va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


#TTx
# 3. Average images uploaded per patient
# Count number of commas in each cell +1
def count_images(cell):
    if pd.isna(cell) or cell == '#N/A':
        return 0
    return cell.count(',') + 1

df['Num_images'] = df['Images'].apply(count_images)
average_images_per_patient = df['Num_images'].mean()
print(f"Average images uploaded per patient: {average_images_per_patient:.2f}")



# =========================
# Pie chart: Image vs No Image
# =========================
# column 'Images'
# Check if cell contains 'http'
df['Has_image'] = df['Images'].str.contains('http', na=False)
image_counts = df['Has_image'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(image_counts, labels=['No Image','Has Image'], autopct='%1.1f%%', colors=['lightgray','red'])
plt.title("Distribution of Rows with Images vs No Images")
plt.show()



# =========================
# Bar plot: Patients with Images vs No Images (with total rows)
# =========================
image_bar_counts = df['Has_image'].value_counts().rename({True: 'Has Image', False: 'No Image'})
total_rows = len(df)

plt.figure(figsize=(6,4))
bars = plt.bar(image_bar_counts.index, image_bar_counts.values, color='red')
plt.xlabel("Image Availability")
plt.ylabel("Number of Patients")
plt.title(f"Patients with vs without Images (Total Rows = {total_rows})")

# Add data labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, str(int(height)), ha='center')

plt.tight_layout()
plt.show()

print(f"Total rows in CSV: {total_rows}")

#--

# =========================
# Subset: Patients WITH Images
# =========================
df_with_images = df[df['Has_image'] == True]

single_with_images = df_with_images[df_with_images['Diagnosis_type'] == 'Single']['Primary & Provisional'].value_counts().reset_index()
single_with_images.columns = ['Diagnosis', 'Count']

multiple_with_images = df_with_images[df_with_images['Diagnosis_type'] == 'Multiple']['Primary & Provisional'].value_counts().reset_index()
multiple_with_images.columns = ['Diagnosis', 'Count']

# Top 20 bar plots
plt.figure(figsize=(10,6))
top20_single_with_images = single_with_images.head(20)
bars = plt.barh(top20_single_with_images['Diagnosis'], top20_single_with_images['Count'], color='red')
plt.xlabel("Frequency")
plt.ylabel("Single Diagnosis (With Images)")
plt.title("Top 20 Single Diagnoses - Patients with Images")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, str(int(width)), va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
top20_multiple_with_images = multiple_with_images.head(20)
bars = plt.barh(top20_multiple_with_images['Diagnosis'], top20_multiple_with_images['Count'], color='red')
plt.xlabel("Frequency")
plt.ylabel("Multiple Diagnosis (With Images)")
plt.title("Top 20 Multiple Diagnoses - Patients with Images")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, str(int(width)), va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# =========================
# Subset: Patients WITHOUT Images
# =========================
df_no_images = df[df['Has_image'] == False]

single_no_images = df_no_images[df_no_images['Diagnosis_type'] == 'Single']['Primary & Provisional'].value_counts().reset_index()
single_no_images.columns = ['Diagnosis', 'Count']

multiple_no_images = df_no_images[df_no_images['Diagnosis_type'] == 'Multiple']['Primary & Provisional'].value_counts().reset_index()
multiple_no_images.columns = ['Diagnosis', 'Count']

# Top 20 bar plots
plt.figure(figsize=(10,6))
top20_single_no_images = single_no_images.head(20)
bars = plt.barh(top20_single_no_images['Diagnosis'], top20_single_no_images['Count'], color='red')
plt.xlabel("Frequency")
plt.ylabel("Single Diagnosis (No Images)")
plt.title("Top 20 Single Diagnoses - Patients without Images")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, str(int(width)), va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
top20_multiple_no_images = multiple_no_images.head(20)
bars = plt.barh(top20_multiple_no_images['Diagnosis'], top20_multiple_no_images['Count'], color='red')
plt.xlabel("Frequency")
plt.ylabel("Multiple Diagnosis (No Images)")
plt.title("Top 20 Multiple Diagnoses - Patients without Images")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, str(int(width)), va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
# =========================
# Number of unique medications
# =========================
# column 'Medicines'
# Function to split medicines correctly
def split_medicines(row):
    if pd.isna(row):
        return []
    # Split on commas that are NOT inside parentheses
    parts = re.split(r',(?![^()]*\))', row)
    return [p.strip().rstrip('.') for p in parts if p.strip()]

# Apply splitting
df['Medicines_split'] = df['Medicines'].apply(split_medicines)

# Flatten all medicines
all_meds = df['Medicines_split'].explode().reset_index(drop=True)

# Frequency table
med_freq = all_meds.value_counts().reset_index()
med_freq.columns = ['Medicine', 'Count']

# Save full frequency table for Excel review
med_freq.to_csv("medicine_frequency_table.csv", index=False)
print("Full medicine frequency table saved as 'medicine_frequency_table.csv'")

# ------------------------
# 1. Plot Top 20 Medicines
# ------------------------
top20 = med_freq.head(20)
plt.figure(figsize=(10,6))
bars = plt.barh(top20['Medicine'], top20['Count'], color='red')
plt.xlabel("Frequency")
plt.ylabel("Medicine")
plt.title("Top 20 Prescribed Medicines")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, str(int(width)), va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ------------------------
# 2. Plot All Medicines
# ------------------------
plt.figure(figsize=(12,10))
bars = plt.barh(med_freq['Medicine'], med_freq['Count'], color='red')
plt.xlabel("Frequency")
plt.ylabel("Medicine")
plt.title("All Prescribed Medicines Frequency")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, str(int(width)), va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ------------------------
# Optional: Top 20 Bar plot
# ------------------------
import matplotlib.pyplot as plt
top20 = med_freq.head(20)

plt.figure(figsize=(10,6))
bars = plt.barh(top20['Medicine'], top20['Count'], color='red')
plt.xlabel("Frequency")
plt.ylabel("Medicine")
plt.title("Top 20 Prescribed Medicines")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, str(int(width)), va='center')
plt.gca().invert_yaxis()  # Largest at top
plt.tight_layout()
plt.show()


# =====================
# Bar chart (Top 20)
# =====================
top20 = med_freq.head(20)

plt.figure(figsize=(10,6))
bars = plt.barh(top20['Medicine'], top20['Count'], color='red')
plt.xlabel("Frequency")
plt.ylabel("Medicine")
plt.title("Top 20 Prescribed Medicines")

# Add data labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, str(int(width)), va='center')

plt.gca().invert_yaxis()  # Largest at top
plt.tight_layout()
plt.show()


# =========================
# Number of referrals (PHC/DH/SDH)
# =========================
# column is 'Referral_advice'
# Count frequency
phc_count = df['Referral_advice'].str.contains('PHC', na=False).sum()
dh_count = df['Referral_advice'].str.contains('DH', na=False).sum()
sdh_count = df['Referral_advice'].str.contains('SDH', na=False).sum()

referral_counts = pd.DataFrame({
    'Facility': ['PHC','DH','SDH'],
    'Count': [phc_count, dh_count, sdh_count]
})

# Bar chart
plt.figure(figsize=(6,4))
bars = plt.bar(referral_counts['Facility'], referral_counts['Count'], color='red')
plt.xlabel("Facility")
plt.ylabel("Number of Referrals")
plt.title("Referrals Suggested by Facility")

# Add data labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, str(int(height)), ha='center')

plt.tight_layout()
plt.show()

# Save referral counts
referral_counts.to_csv("referral_counts.csv", index=False)

#DATES
# Ensure datetime format
df['Visit_started_date'] = pd.to_datetime(df['Visit_started_date'], errors='coerce')

# Drop missing dates
dates = df['Visit_started_date'].dropna()

# ------------------------
# 1. Plot visits per day
# ------------------------
daily_counts = dates.value_counts().sort_index()  # sort by date
# Print summary stats
# ------------------------
start_date = dates.min()
end_date = dates.max()
total_days = (end_date - start_date).days + 1
total_visits = len(dates)

print(f"Visit date range: {start_date.date()} to {end_date.date()} ({total_days} days)")
print(f"Total visits in CSV: {total_visits}")


# Ensure datetime format
df['Visit_started_date'] = pd.to_datetime(df['Visit_started_date'], errors='coerce')
df['Month'] = df['Visit_started_date'].dt.to_period('M')

# ------------------------
# 1. Gender stratification by month
# ------------------------
gender_month = df.groupby(['Month', 'Gender']).size().unstack(fill_value=0)
print("Gender stratification by month:")
print(gender_month)

# Plot gender stratification
gender_month.plot(kind='bar', figsize=(10,6), color=['red','lightblue'])
plt.xlabel("Month")
plt.ylabel("Number of Patients")
plt.title("Gender Stratification of Patients by Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------
# 2. Age stratification by month
# ------------------------
# Define age groups (customizable)
bins = [0, 12, 18, 35, 50, 65, 100]
labels = ['0-12','13-18','19-35','36-50','51-65','66+']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

age_month = df.groupby(['Month','Age_Group']).size().unstack(fill_value=0)
print("Age stratification by month:")
print(age_month)

# Plot age stratification
age_month.plot(kind='bar', stacked=True, figsize=(12,6), colormap='Reds')
plt.xlabel("Month")
plt.ylabel("Number of Patients")
plt.title("Age Group Stratification of Patients by Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



