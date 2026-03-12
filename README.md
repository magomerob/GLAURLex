# GLAURLex

<div align="center">
  <img src="logo_glaur.png" alt="GLAURLex Logo" width="300"/>
</div>

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd GLAURLex

# Install dependencies
poetry install

# Run the application
poetry run streamlit run src/glaurlex/ui/app.py
```

The application will open at `http://localhost:8501`

### Building Documentation
```bash
doxygen Doxyfile
```

## Deployment

### Docker
```bash
# Build Docker image
docker build -t glaurlex:latest .

# Run container
docker run -p 8501:8501 glaurlex:latest
```

Edit `src/glaurlex/config.py` to customize:
- Default data directories
- Processing parameters
- UI settings
- Graph analysis options

## License

See LICENSE file for details.
