# ECG Insight AI

A modern, production-ready React + Vite frontend for AI-powered ECG analysis system.

## 🚀 Features

- **Modern UI**: Clean, medical-grade design with healthcare-focused aesthetics
- **File Upload**: Drag-and-drop support for 6-lead Kardia ECG files (PNG, JPG, PDF)
- **ECG Viewer**: Interactive viewer with zoom and scroll capabilities
- **AI Analysis**: Real-time arrhythmia detection with confidence scores
- **Responsive**: Fully responsive design for desktop and mobile
- **Accessibility**: WCAG-compliant with proper contrast and navigation

## 🛠️ Tech Stack

- **Framework**: React 18 with Vite
- **Styling**: Tailwind CSS with custom medical color palette
- **Icons**: Lucide React
- **Animations**: Framer Motion
- **Routing**: React Router DOM
- **PDF Viewing**: React PDF (planned)
- **Image Zoom**: React Medium Image Zoom (planned)

## 🎨 Design System

### Color Palette
- Primary: `#0EA5A4` (Teal)
- Secondary: `#2563EB` (Soft Medical Blue)
- Background: `#F8FAFC`
- Card: `#FFFFFF`
- Text Primary: `#0F172A`
- Text Secondary: `#475569`
- Border: `#E2E8F0`
- Success: `#22C55E`
- Warning: `#F59E0B`

### Typography
- Font Family: Inter (Roboto, system-ui fallback)
- Headings: Semi-bold
- Body: Regular with high readability

## 📁 Project Structure

```
src/
├── components/
│   ├── Navbar.jsx           # Top navigation
│   ├── HeroSection.jsx      # Landing page hero
│   ├── UploadCard.jsx       # File upload component
│   ├── SampleECG.jsx        # Sample ECG preview
│   ├── ECGViewer.jsx        # ECG file viewer
│   ├── ResultCard.jsx       # Analysis results display
│   └── Disclaimer.jsx       # Medical disclaimer
├── pages/
│   ├── Home.jsx            # Landing page
│   ├── Analyze.jsx         # Upload and analysis page
│   └── Results.jsx         # Results display page
├── utils/
│   └── fileValidation.js   # File validation utilities
├── App.jsx                 # Main app component
├── main.jsx               # App entry point
└── index.css              # Global styles
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ecg-insight-ai/frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Open in browser**
   ```
   http://localhost:3001
   ```

### Backend Integration

The frontend expects a backend API running on `http://localhost:8000` with the following endpoint:

```
POST /predict
Content-Type: multipart/form-data
Body: file (ECG image/PDF)
```

Response format:
```json
{
  "predicted_class": "Normal Sinus Rhythm",
  "confidence": 92,
  "probabilities": {
    "Normal Sinus Rhythm": 92,
    "Atrial Fibrillation": 3,
    "Ventricular Tachycardia": 2,
    "Bradycardia": 2,
    "Premature Ventricular Contraction": 1
  },
  "quality": "Good"
}
```

## 📱 Usage

1. **Home Page**: View hero section and sample ECG options
2. **Upload ECG**: Drag and drop or browse for 6-lead Kardia ECG files
3. **View Results**: See AI analysis with confidence scores and probabilities
4. **ECG Viewer**: Zoom and inspect ECG waveforms

## 🎯 Key Features

### File Upload
- Drag & drop interface
- File type validation (PNG, JPG, PDF)
- Size limit: 10MB
- Progress indicators

### ECG Analysis
- Real-time AI processing
- Confidence score display
- Probability breakdown for all classes
- Signal quality assessment

### Medical Compliance
- Healthcare-focused design
- Clear medical disclaimers
- Research/educational purpose labeling
- Professional typography

## 🔧 Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Environment Variables

Create a `.env` file in the root directory:

```env
VITE_API_URL=http://localhost:8000
```

## 📋 TODO / Future Enhancements

- [ ] Integrate React PDF for PDF viewing
- [ ] Add React Medium Image Zoom for image zoom
- [ ] Implement real backend API integration
- [ ] Add user authentication
- [ ] Add report download functionality
- [ ] Add ECG waveform visualization
- [ ] Add comparison tools
- [ ] Add historical analysis tracking

## ⚠️ Medical Disclaimer

This application is for research and educational purposes only. It does not replace professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📞 Support

For questions or support, please open an issue in the repository.