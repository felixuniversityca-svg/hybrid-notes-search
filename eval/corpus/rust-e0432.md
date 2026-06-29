# Rust error E0432

E0432 is the unresolved-import error. The compiler raises it when a `use`
declaration points at a path that does not exist, for example importing from a
crate you forgot to add to Cargo.toml, or naming a module that is not declared
with `mod`. The fix is to correct the path, add the missing dependency, or
declare the module so the import resolves.
