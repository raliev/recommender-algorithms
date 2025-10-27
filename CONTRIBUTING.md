# Contributing to Recommender System Laboratory

Thank you for your interest in contributing to the Recommender System Laboratory! This document provides guidelines and instructions for contributing.

## 🎯 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python and package versions
- Any relevant screenshots or error messages

### Suggesting Features

We welcome feature suggestions! Please create an issue describing:
- The feature and its benefits
- Use cases and examples
- Any implementation ideas

### Adding New Algorithms

To add a new recommendation algorithm:

1. **Implement the algorithm** in `algorithms/`
   - Inherit from `Recommender` base class
   - Implement `fit()` and `predict()` methods
   - Add to `algorithms/__init__.py`

2. **Configure the algorithm** in `algorithm_config.py`
   - Add algorithm parameters
   - Define hyperparameter widgets
   - Configure visualization classes

3. **Add visualizations** in `visualization/`
   - Create a visualizer class
   - Create a renderer class
   - Add visualization documentation

4. **Test your implementation**
   - Test with different datasets
   - Verify metrics calculation
   - Check visualization output

### Improving Documentation

Help us improve:
- README clarity
- Algorithm documentation
- Code comments and docstrings
- Tutorial examples

### Code Style

- Follow PEP 8 Python style guide
- Use meaningful variable names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Add comments for complex logic

## 🔄 Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### PR Guidelines

- **Clear Title**: Summarize your changes concisely
- **Description**: Explain what and why, not just what
- **Keep PRs Focused**: One feature or fix per PR
- **Test Coverage**: Include tests for new features
- **Update Docs**: Update README or docs if needed

## 📝 Code Review Process

All submissions require review. We aim to review PRs within a week. You may receive feedback and requests for changes. Please respond constructively.

## 🎓 Learning Resources

- Read the book: [Recommender Algorithms in 2026](https://testmysearch.com/books/recommender-algorithms.html)
- Streamlit docs: [streamlit.io/docs](https://docs.streamlit.io)
- Python style guide: [PEP 8](https://www.python.org/dev/peps/pep-0008/)

## 📧 Questions?

Feel free to open an issue with the `question` label for clarifications or discussions.

Thank you for contributing! 🎉

