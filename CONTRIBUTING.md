# Contributing to CatalogMatch

Thank you for your interest in contributing to CatalogMatch! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Windows 10+ (for desktop app testing)
- Basic knowledge of Python, Flask, and OpenCV

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/g1mliii/image-match.git
   cd image-match
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## Project Structure

```
image-match/
├── backend/           # Python backend (Flask API, matching engine)
│   ├── api/          # REST API endpoints
│   ├── core/         # Core matching logic
│   ├── database/     # Database models and access
│   └── utils/        # Utility functions
├── docs/             # GitHub Pages website
├── test_data/        # Sample test data
├── main.py          # Desktop app launcher
└── requirements.txt  # Python dependencies
```

## Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

**When reporting bugs, include:**
- Operating system and version
- Python version
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- Error messages or logs

**Create an issue:** https://github.com/g1mliii/image-match/issues/new

## Suggesting Features

We welcome feature suggestions! Please:
1. Check if the feature has already been requested
2. Clearly describe the feature and its use case
3. Explain why it would be valuable
4. Provide examples if possible

## Pull Request Process

### Before Submitting

1. **Create an issue** first to discuss major changes
2. **Fork the repository** and create a feature branch
3. **Follow code style** (PEP 8 for Python)
4. **Test your changes** thoroughly
5. **Update documentation** if needed

### Submitting a PR

1. **Branch naming:**
   - `feature/description` for new features
   - `fix/description` for bug fixes
   - `docs/description` for documentation

2. **Commit messages:**
   - Use clear, descriptive messages
   - Start with a verb (Add, Fix, Update, Remove)
   - Example: "Add GPU acceleration for image preprocessing"

3. **PR description:**
   - Reference related issues (#123)
   - Describe what changed and why
   - Include screenshots for UI changes
   - List any breaking changes

4. **Code review:**
   - Address review comments promptly
   - Keep PRs focused and reasonably sized
   - Be open to feedback

## Code Style

### Python
- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions small and focused

```python
def calculate_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate visual similarity between two images.
    
    Args:
        image1: First image as numpy array
        image2: Second image as numpy array
        
    Returns:
        Similarity score between 0 and 100
    """
    # Implementation
    pass
```

### JavaScript (Website)
- Use ES6+ features
- Use meaningful variable names
- Add comments for complex logic
- Keep functions pure when possible

## Testing

- Write tests for new features
- Ensure existing tests pass
- Test with real-world data
- Test edge cases (empty inputs, large files, etc.)

## Documentation

- Update README.md for user-facing changes
- Update code comments for implementation changes
- Update website docs for feature additions
- Keep documentation clear and concise

## Development Guidelines

### Adding New Features

1. **Plan first** - Discuss in an issue
2. **Small PRs** - Break large features into smaller PRs
3. **Test thoroughly** - Include edge cases
4. **Document** - Update relevant docs
5. **Performance** - Consider impact on performance

### Performance Considerations

- Profile code before optimizing
- Use GPU acceleration where beneficial
- Cache expensive computations
- Optimize database queries
- Handle large batches efficiently

### Error Handling

- Provide clear error messages
- Handle edge cases gracefully
- Log errors appropriately
- Don't expose sensitive information

## Security

- Never commit sensitive data (API keys, passwords)
- Validate all user inputs
- Sanitize file uploads
- Follow security best practices

**Report security issues privately to:** info@anchored.site

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

## Communication

- **GitHub Issues** - Bug reports and feature requests
- **Pull Requests** - Code contributions and discussions
- **Email** - info@anchored.site for private matters

## Priority Areas

We're especially interested in contributions for:

- [ ] GPU acceleration improvements
- [ ] macOS support
- [ ] Performance optimizations
- [ ] Additional image formats
- [ ] UI/UX improvements
- [ ] Documentation improvements
- [ ] Test coverage

## Resources

- [Python Documentation](https://docs.python.org/3/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Git Workflow Guide](https://guides.github.com/introduction/flow/)

## Questions?

If you have questions about contributing:
1. Check existing issues and discussions
2. Read the documentation
3. Open a new issue with the "question" label
4. Email info@anchored.site

---

**Thank you for contributing to CatalogMatch!**

Every contribution, no matter how small, helps make this project better.
