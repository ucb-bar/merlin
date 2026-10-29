All three native commands (compile, transitive audit, exact sixteen root types)
returned0. The Python wrapper then failed because it compared the top-level
`{"statements": ...}` object keys with root names instead of its nested map.
Original script/raw outputs/rc1 log are preserved. bind-retained.py corrects only
the OFFLINE output-schema access and binds those same successful native records.
No native command was repeated and no proof/source/expected pin was changed.
