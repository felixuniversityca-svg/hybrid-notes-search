# Rust error E0433

E0433 is the failed-to-resolve error. The compiler raises it when you reference a
name that is not in scope, such as calling a type or function through a path that
was never brought into scope, or a simple typo in the path. The fix is usually to
add the right `use` import, correct the spelling, or fully qualify the path.
