# Verification test witnesses

Own only the authored source programs, fixture classes, controlled processes and reference
data used by the verification tests. Keep the actual test bodies in the existing test
buckets; do not hide tests here or import retired test modules.

Import production verifiers from `merlin.verification`. Keep verification outcomes,
independent numerical references and unavailable prerequisites distinct; these witnesses
cannot redefine acceptance policy or establish hardware execution evidence.
