# Contributing to Motion Tracker

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/MindDock/motion-tracker.git
   cd motion-tracker
   ```
3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write clean, documented code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
pytest tests/
```

### 4. Format Code

```bash
black src/ demos/ tests/
flake8 src/ demos/ tests/
```

### 5. Commit Changes

```bash
git add .
git commit -m "Description of changes"
```

Use clear, descriptive commit messages:
- `feat: Add YOLO11 backend support`
- `fix: Correct angle calculation for acute angles`
- `docs: Update installation instructions`
- `refactor: Simplify motion analyzer logic`

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## What to Contribute

### High-Priority Areas

1. **New Backends:**
   - Apple Vision Framework implementation
   - YOLO11 pose backend
   - Custom model integration

2. **Applications:**
   - Dance coach demo
   - Sports analysis tools
   - Rehabilitation exercises

3. **Features:**
   - Multi-person tracking
   - 3D reconstruction
   - AR overlays
   - Export functionality

4. **Documentation:**
   - API reference
   - Tutorials
   - Example code
   - Video demonstrations

5. **Testing:**
   - Unit tests
   - Integration tests
   - Performance benchmarks

### Good First Issues

Look for issues labeled `good-first-issue`:
- Documentation improvements
- Code formatting
- Adding examples
- Bug fixes

## Code Style

### Python Style Guide

Follow PEP 8 with these specifics:

```python
# Good
def calculate_angle(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray
) -> float:
    """Calculate angle formed by three points.

    Args:
        point_a: First point
        point_b: Vertex point
        point_c: Third point

    Returns:
        Angle in degrees
    """
    # Implementation
    pass

# Use type hints
# Document all public functions
# Keep functions focused and short
```

### Naming Conventions

- Classes: `PascalCase` (e.g., `PoseEstimator`)
- Functions/methods: `snake_case` (e.g., `calculate_angle`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_KEYPOINTS`)
- Private: `_leading_underscore` (e.g., `_internal_method`)

## Testing Guidelines

### Writing Tests

```python
import pytest
from src.core.angle_calculator import AngleCalculator

def test_angle_calculation():
    """Test basic angle calculation."""
    calculator = AngleCalculator()

    # Test right angle
    a = np.array([0, 0, 0])
    b = np.array([1, 0, 0])
    c = np.array([1, 1, 0])

    angle = calculator.calculate_angle_3points(a, b, c)
    assert abs(angle - 90.0) < 0.1
```

### Test Coverage

Aim for:
- Core algorithms: 90%+ coverage
- Backends: 70%+ coverage
- Applications: 50%+ coverage

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def process_frame(self, frame: np.ndarray) -> Optional[PoseResult]:
    """Process a single frame and detect pose.

    This method performs pose estimation on the input frame using
    the configured backend.

    Args:
        frame: Input image as numpy array (BGR format)

    Returns:
        PoseResult object if detection successful, None otherwise

    Raises:
        ValueError: If frame is invalid

    Example:
        >>> estimator = MediaPipeBackend()
        >>> estimator.initialize()
        >>> result = estimator.process_frame(frame)
    """
```

### README Updates

When adding features:
1. Update main README.md
2. Add to CHANGELOG.md
3. Update relevant documentation

## Pull Request Process

### Before Submitting

- [ ] Tests pass
- [ ] Code is formatted
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Descriptive PR title

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Tests pass
- [ ] Code formatted
- [ ] Documentation updated
```

### Review Process

1. Maintainers review within 1-2 weeks
2. Address feedback
3. Approval from at least one maintainer
4. Merge

## Adding a New Backend

Template for new backend:

```python
from src.core.pose_estimator import PoseEstimator, PoseResult

class YourBackend(PoseEstimator):
    """Your backend description."""

    def initialize(self) -> bool:
        """Initialize backend."""
        # Implementation
        pass

    def process_frame(self, frame: np.ndarray) -> Optional[PoseResult]:
        """Process frame."""
        # Implementation
        pass

    def release(self):
        """Cleanup."""
        # Implementation
        pass

    def get_keypoint_names(self) -> List[str]:
        """Get keypoint names."""
        # Implementation
        pass

    @property
    def backend_name(self) -> str:
        """Backend name."""
        return "YourBackend"
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open an issue for bugs or feature requests
- Join our Discord for discussions
- Email maintainers for private inquiries

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing! 🙏
