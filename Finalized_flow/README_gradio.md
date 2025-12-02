# 🔍 OCR Processing Suite - Gradio UI

A comprehensive web interface for Vision Language Model (VLM) based OCR processing, specifically designed for Nepali student forms.

## 🚀 Quick Start

### Method 1: Using the Launch Script
```bash
cd /home/museum/Fine-tuned-TrOCR/Finalized_flow
./launch_gradio.sh
```

### Method 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements_gradio.txt

# Launch the simplified UI
python gradio_ui_simplified.py

# Or launch the full-featured UI
python gradio_ui.py
```

## 📋 Features

### 1. 📄 Single Image OCR
- **Purpose**: Extract structured data from individual images
- **Input**: Single image file (PNG, JPG, JPEG, etc.)
- **Output**: JSON formatted extraction results
- **Use Case**: Quick testing or processing single forms

**How to Use:**
1. Upload an image using the file uploader
2. Click "🚀 Process Image"
3. View extracted data in JSON format
4. Check processing status and any error messages

### 2. 📁 Batch Processing
- **Purpose**: Process multiple images simultaneously
- **Input**: Multiple image files
- **Output**: JSON results + downloadable CSV file
- **Use Case**: Processing large datasets efficiently

**How to Use:**
1. Select multiple images using the file uploader
2. Click "🔄 Process All Images"
3. Monitor processing progress
4. Download CSV file with all results

### 3. 📱 Phone Number Localization
- **Purpose**: Automatically detect and crop phone number regions
- **Input**: Image containing phone numbers
- **Output**: Image with bounding box + cropped phone number region
- **Use Case**: Focused OCR on specific fields, data validation

**How to Use:**
1. Upload an image containing phone numbers
2. Click "🎯 Localize Phone Numbers"
3. View the detected bounding box overlay
4. Download the cropped phone number region

### 4. 📊 CSV Analysis
- **Purpose**: Calculate accuracy metrics from OCR results
- **Input**: CSV file with OCR results
- **Output**: Accuracy metrics table and summary statistics
- **Use Case**: Model evaluation, quality assessment

**How to Use:**
1. Upload a CSV file with OCR results
2. Set row limit (0 for all rows)
3. Click "📈 Analyze CSV"
4. Review accuracy metrics and summary

### 5. 🔄 CSV Comparison
- **Purpose**: Compare results from two different models/runs
- **Input**: Two CSV files
- **Output**: Difference report CSV and comparison summary
- **Use Case**: Model comparison, A/B testing

**How to Use:**
1. Upload two CSV files to compare
2. Click "⚖️ Compare Files"
3. Download the difference report
4. Review comparison summary statistics

### 6. 🌐 HTML Report Generation
- **Purpose**: Create visual HTML reports for easy sharing
- **Input**: CSV file with OCR results
- **Output**: Self-contained HTML report
- **Use Case**: Presenting results, documentation

**How to Use:**
1. Upload a CSV file with results
2. Click "📄 Generate HTML Report"
3. Download the HTML file
4. Open in any web browser for viewing

## 🛠️ Technical Details

### Supported Models
- **MiniCPM-V-4_5**: Primary VLM for OCR extraction
- **Qwen-VL**: Alternative VLM (requires setup)
- **PaddleOCR**: Traditional OCR backend (optional)

### Field Extraction
The system extracts the following fields from Nepali student forms:
- Name
- Class
- Address  
- Faculty
- Phone Number
- Guardian's Number
- Email
- Interested Course
- School Name
- Who Gave Seminar
- Rate Us

### Validation Rules
- **Phone Numbers**: Must be 10 digits, start with 97/98
- **Email**: Must contain '@' and valid domain
- **Class**: Converts Roman numerals to Arabic numbers
- **Address**: Standardizes common Nepali abbreviations

## 🔧 Configuration

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU selection
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory optimization
```

### Model Loading
Models are cached after first load to improve performance. The system will automatically download and cache models on first use.

## 📊 Output Formats

### JSON Output (Single Image)
```json
{
  "extracted_data": {
    "Name": "राम बहादुर",
    "class": "12",
    "phone number": "9801234567",
    "email": "ram@gmail.com",
    // ... other fields
  },
  "processing_time": 3.45,
  "timestamp": "2024-12-02 10:30:00"
}
```

### CSV Output (Batch Processing)
| file_name | Name | phone number | email | class | ... |
|-----------|------|--------------|-------|-------|-----|
| image1.jpg | राम बहादुर | 9801234567 | ram@gmail.com | 12 | ... |
| image2.jpg | सीता देवी | 9807654321 | sita@gmail.com | 11 | ... |

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure sufficient GPU memory (>8GB recommended)
   - Check CUDA installation
   - Try CPU inference by modifying model loading code

2. **Out of Memory**
   - Reduce batch size
   - Process images individually
   - Set environment variable: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256`

3. **JSON Parsing Errors**
   - Check if model response contains valid JSON
   - Review raw response in single image processing
   - Consider adjusting prompt for better structured output

4. **Poor Extraction Quality**
   - Ensure images are clear and well-lit
   - Check image resolution (minimum 300 DPI recommended)
   - Verify handwriting legibility

### Performance Optimization

- **GPU Usage**: Recommended 8GB+ VRAM
- **RAM**: Minimum 16GB for batch processing
- **Storage**: Models require ~15GB disk space
- **Network**: Fast internet for initial model download

## 📁 File Structure

```
Finalized_flow/
├── gradio_ui.py                 # Full-featured UI
├── gradio_ui_simplified.py      # Simplified UI (recommended)
├── requirements_gradio.txt      # Dependencies
├── launch_gradio.sh            # Launch script
├── README_gradio.md            # This file
└── [existing OCR modules]      # Core functionality
```

## 🤝 Contributing

To extend functionality:
1. Add new functions to `gradio_ui_simplified.py`
2. Import existing modules from the Finalized_flow directory
3. Create new Gradio components and connect them
4. Update this README with new features

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review console output for error messages  
3. Ensure all dependencies are installed
4. Verify model accessibility and GPU availability

## 🎯 Best Practices

1. **Image Quality**: Use clear, well-lit images with good contrast
2. **Batch Size**: Process 10-50 images per batch for optimal performance
3. **Validation**: Always review extracted data, especially phone numbers and emails
4. **Backup**: Keep original images and results for reference
5. **Testing**: Test with sample images before processing large datasets

---

**Note**: This UI integrates all functionality from the Finalized_flow folder into a user-friendly web interface. The underlying OCR models and processing logic remain unchanged, ensuring consistent results with existing workflows.