# CLAUDE.md - Modular Design Guidelines

# With the message from the user, the database will first be checked. If there is any relevant content, it will be fetched. If not, a search will be made with the API and the relevant result or results will be returned and the user will be responded to. The content fetched with the API will be added to the database.

## Package Manager
- **uv**: Use uv for all Python package management
- `uv add <package>`: Add dependencies
- `uv remove <package>`: Remove dependencies
- `uv sync`: Sync environment with lock file
- `uv run <command>`: Run commands in virtual environment

## Core Design Principles

### 1. KISS (Keep It Simple, Stupid)
- **ALWAYS** choose the simplest solution that works
- Prefer straightforward approaches over clever/complex ones
- Simple code is easier to understand, maintain, and debug
- If you can't explain it simply, it's probably too complex

### 2. YAGNI (You Aren't Gonna Need It)
- **NEVER** build functionality on speculation
- Implement features ONLY when they are actually needed
- Avoid over-engineering for hypothetical future requirements
- Focus on current, real requirements

### 3. Single Responsibility Principle
- Each module should have ONE clear purpose
- Functions should do ONE thing well
- Classes should represent ONE concept
- Files should contain related functionality only

### 4. Separation of Concerns
- **Data Layer**: Models, schemas, database interactions
- **Business Logic**: Core application logic, algorithms
- **Presentation Layer**: UI components, templates, views
- **Configuration**: Settings, environment variables
- **Utilities**: Helper functions, common tools

### 5. Dependency Management
- Use dependency injection patterns
- Minimize coupling between modules
- Define clear interfaces/contracts
- Avoid circular dependencies

### 6. Modular Design
- Never forget we are designing based modular


## File Organization Structure

```
project/
├── src/
│   ├── models/          # Data models and schemas
│   ├── services/        # Business logic services
│   ├── controllers/     # Request handlers
│   ├── utils/           # Utility functions
│   ├── config/          # Configuration files
│   ├── interfaces/      # Type definitions/interfaces
│   └── tests/           # Test files mirroring src structure
├── docs/                # Documentation
└── scripts/             # Build/deployment scripts
```

## Modular Code Patterns

### Module Export Patterns
- Use named exports for multiple utilities
- Use default exports for main module functionality
- Create index.js files for clean imports
- Group related exports in barrel files

### Interface Design
- Define clear input/output contracts
- Use TypeScript interfaces or JSDoc for documentation
- Validate inputs at module boundaries
- Return consistent data structures

### Error Handling
- Each module handles its own errors
- Use consistent error types across modules
- Propagate errors with clear context
- Implement proper logging per module



## Code Style Guidelines

### Naming Conventions
- Use descriptive, self-documenting names
- Follow language-specific conventions
- Be consistent across all modules
- Use verbs for functions, nouns for variables

### Documentation
- Document module purpose and usage
- Include examples for complex modules
- Maintain API documentation
- Comment non-obvious business logic

## Commands and Workflows

### Development Commands
- `uv run dev`: Start development server
- `uv run lint`: Check code style
- `uv run format`: Format code
- `uv run build`: Build for production
- `uv sync`: Sync dependencies

### Module Creation Workflow
1. Define module interface first
2. Write tests for expected behavior
3. Implement module functionality
4. Add documentation and examples
5. Update main module index

## Best Practices

### IMPORTANT Guidelines
- **YOU MUST** create interfaces before implementation
- **ALWAYS** write tests for new modules
- **NEVER** directly import from deep nested paths
- **PREFER** composition over inheritance
- **USE** dependency injection for external services
- **GIVE** little examples

### Code Review Checklist
- [ ] Module has single responsibility
- [ ] Clear interface definition
- [ ] Proper error handling
- [ ] Unit tests included
- [ ] Documentation updated
- [ ] No circular dependencies
- [ ] Follows naming conventions

## Anti-Patterns to Avoid
- God objects/modules doing too much
- Tight coupling between unrelated modules
- Shared mutable state
- Deep inheritance hierarchies
- Circular dependencies
- Magic numbers/strings without constants

## Module Communication
- Use events for loose coupling
- Implement observer patterns for notifications
- Use message passing for async operations
- Define clear data contracts between modules
- Avoid direct module manipulation

## Configuration Management
- Environment-specific configs in separate files
- Use configuration validation
- Centralize configuration access
- Support configuration overrides
- Document all configuration options

## Performance Considerations
- Lazy load modules when possible
- Cache expensive module operations
- Profile module performance regularly
- Optimize module loading paths
- Monitor memory usage per module