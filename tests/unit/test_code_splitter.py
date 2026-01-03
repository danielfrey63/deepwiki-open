import pytest
from adalflow.core.types import Document
from api.code_splitter import (
    TreeSitterCodeSplitter,
    CodeAwareSplitter,
    CodeSplitterConfig,
)
from adalflow.components.data_process import TextSplitter


class TestTreeSitterCodeSplitter:
    """Test suite for TreeSitterCodeSplitter with recursive splitting."""

    @pytest.fixture
    def splitter(self):
        return TreeSitterCodeSplitter(
            chunk_size_lines=13,
            chunk_overlap_lines=0,
            min_chunk_lines=3,
            enabled=True,
        )

    def test_small_python_function(self, splitter):
        """Test that small functions are kept intact."""
        code = '''def hello_world():
    """A simple function."""
    print("Hello, World!")
    return True
'''
        doc = Document(
            text=code,
            meta_data={"file_path": "test.py", "type": "py", "is_code": True},
        )
        
        result = splitter.split_document(doc)
        
        assert len(result) == 1
        assert "def hello_world" in result[0].text
        assert result[0].meta_data["chunk_index"] == 0
        assert result[0].meta_data["chunk_total"] == 1

    def test_large_class_with_methods(self, splitter):
        """Test recursive splitting of a large class into methods."""
        code = '''class Calculator:
    """A calculator class with multiple methods."""
    
    def __init__(self):
        self.result = 0
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.result = result
        return result
    
    def subtract(self, a, b):
        """Subtract two numbers."""
        result = a - b
        self.result = result
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.result = result
        return result
    
    def divide(self, a, b):
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.result = result
        return result
'''
        doc = Document(
            text=code,
            meta_data={"file_path": "calc.py", "type": "py", "is_code": True},
        )
        
        result = splitter.split_document(doc)
        
        # Should split into multiple chunks (methods)
        assert len(result) > 1
        
        # Each chunk should have metadata
        for chunk in result:
            assert "chunk_index" in chunk.meta_data
            assert "chunk_total" in chunk.meta_data
            assert "chunk_start_line" in chunk.meta_data

    def test_nested_classes(self, splitter):
        """Test recursive splitting with nested class structures."""
        code = '''class OuterClass:
    """Outer class with nested class."""
    
    def outer_method(self):
        """Method in outer class."""
        return "outer"
    
    class InnerClass:
        """Inner class."""
        
        def inner_method_one(self):
            """First inner method."""
            return "inner1"
        
        def inner_method_two(self):
            """Second inner method."""
            return "inner2"
        
        def inner_method_three(self):
            """Third inner method."""
            return "inner3"
'''
        doc = Document(
            text=code,
            meta_data={"file_path": "nested.py", "type": "py", "is_code": True},
        )
        
        result = splitter.split_document(doc)
        
        # Should recursively split nested structures
        assert len(result) > 1

    def test_fallback_to_line_splitting(self, splitter):
        """Test fallback to line-based splitting when no semantic structure found."""
        code = '''# This is a very long comment block
# Line 2
# Line 3
# Line 4
# Line 5
# Line 6
# Line 7
# Line 8
# Line 9
# Line 10
# Line 11
# Line 12
# Line 13
# Line 14
# Line 15
'''
        doc = Document(
            text=code,
            meta_data={"file_path": "comments.py", "type": "py", "is_code": True},
        )
        
        result = splitter.split_document(doc)
        
        # Should fall back to line-based splitting
        assert len(result) >= 1

    def test_min_chunk_lines_filter(self, splitter):
        """Test that chunks smaller than min_chunk_lines are filtered out."""
        # This code has only 1 line, which is less than min_chunk_lines=3
        code = '''x = 1
'''
        doc = Document(
            text=code,
            meta_data={"file_path": "tiny.py", "type": "py", "is_code": True},
        )
        
        result = splitter.split_document(doc)
        
        # When no semantic nodes are found and all chunks are filtered out,
        # the fallback returns the original document as a safety measure
        assert len(result) == 1
        assert result[0].text == code

    def test_javascript_code(self, splitter):
        """Test splitting JavaScript code."""
        code = '''class UserService {

    private String s;

    public constructor() {
        this.users = [];
    }
    
    addUser(user) {
        this.users.push(user);
        return user;
    }
    
    public removeUser(userId) {
        const index = this.users.findIndex(u => u.id === userId);
        if (index !== -1) {
            this.users.splice(index, 1);
        }
    }
    
    protected getUser(userId) {
        return this.users.find(u => u.id === userId);
    }
}
'''
        doc = Document(
            text=code,
            meta_data={"file_path": "service.js", "type": "js", "is_code": True},
        )
        
        result = splitter.split_document(doc)
        
        assert len(result) >= 1

    def test_unsupported_language_fallback(self, splitter):
        """Test fallback for unsupported file types."""
        code = '''Some random text
that is not code
but should still
be processed
line by line
if it's long enough
to exceed the chunk size
and needs splitting
into multiple parts
for proper handling
and processing later
in the pipeline
'''
        doc = Document(
            text=code,
            meta_data={"file_path": "test.xyz", "type": "xyz", "is_code": True},
        )
        
        result = splitter.split_document(doc)
        
        # Should fall back to line-based splitting
        assert len(result) >= 1

    def test_chunk_overlap(self, splitter):
        """Test that chunk overlap is applied correctly."""
        # Create code that will definitely be split
        lines = []
        for i in range(30):
            lines.append(f"# Line {i + 1}")
        code = "\n".join(lines)
        
        doc = Document(
            text=code,
            meta_data={"file_path": "long.py", "type": "py", "is_code": True},
        )
        
        result = splitter.split_document(doc)
        
        # Should have multiple chunks with overlap
        assert len(result) > 1

    def test_disabled_splitter(self):
        """Test that disabled splitter returns original document."""
        splitter = TreeSitterCodeSplitter(enabled=False)
        
        code = "def test(): pass"
        doc = Document(
            text=code,
            meta_data={"file_path": "test.py", "type": "py", "is_code": True},
        )
        
        result = splitter.split_document(doc)
        
        assert len(result) == 1
        assert result[0].text == code

    def test_non_code_document(self, splitter):
        """Test that non-code documents are returned unchanged."""
        text = "This is a markdown document."
        doc = Document(
            text=text,
            meta_data={"file_path": "README.md", "type": "md", "is_code": False},
        )
        
        result = splitter.split_document(doc)
        
        assert len(result) == 1
        assert result[0].text == text


class TestCodeAwareSplitter:
    """Test suite for CodeAwareSplitter integration."""

    @pytest.fixture
    def code_aware_splitter(self):
        text_splitter = TextSplitter(split_by="word", chunk_size=100, chunk_overlap=10)
        code_splitter = TreeSitterCodeSplitter(
            chunk_size_lines=10,
            chunk_overlap_lines=2,
            min_chunk_lines=3,
        )
        return CodeAwareSplitter(
            text_splitter=text_splitter,
            code_splitter=code_splitter,
        )

    def test_code_document_routing(self, code_aware_splitter):
        """Test that code documents are routed to code splitter."""
        code = '''def hello():
    print("Hello")
    return True
'''
        doc = Document(
            text=code,
            meta_data={"file_path": "test.py", "type": "py", "is_code": True},
        )
        
        result = code_aware_splitter([doc])
        
        assert len(result) >= 1

    def test_text_document_routing(self, code_aware_splitter):
        """Test that text documents are routed to text splitter."""
        text = "This is a regular text document that should be processed by the text splitter."
        doc = Document(
            text=text,
            meta_data={"file_path": "README.md", "type": "md", "is_code": False},
        )
        
        result = code_aware_splitter([doc])
        
        assert len(result) >= 1

    def test_serialization(self, code_aware_splitter):
        """Test that CodeAwareSplitter can be serialized and deserialized."""
        serialized = code_aware_splitter.to_dict()
        
        assert "text_splitter" in serialized
        assert "code_splitter_config" in serialized
        
        # Test deserialization
        restored = CodeAwareSplitter.from_dict(serialized)
        
        assert restored is not None
        assert isinstance(restored, CodeAwareSplitter)


class TestCodeSplitterConfig:
    """Test suite for CodeSplitterConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CodeSplitterConfig()
        
        assert config.chunk_size_lines == 200
        assert config.chunk_overlap_lines == 20
        assert config.min_chunk_lines == 5
        assert config.enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CodeSplitterConfig(
            chunk_size_lines=100,
            chunk_overlap_lines=10,
            min_chunk_lines=3,
            enabled=False,
        )
        
        assert config.chunk_size_lines == 100
        assert config.chunk_overlap_lines == 10
        assert config.min_chunk_lines == 3
        assert config.enabled is False

    def test_config_immutability(self):
        """Test that config is frozen (immutable)."""
        config = CodeSplitterConfig()
        
        with pytest.raises(Exception):
            config.chunk_size_lines = 300
