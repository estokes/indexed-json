# Indexed Json

With this module you can store serde serializable types in simple
newline delimited json formatted text files and index fields in
those records for quick querying and retreival of matching records
not unlike in a relational database.

The main intended use case is logging of important books and
records that need to be stored in a simple and accessible format
which can be processed by external tools and backed up online by
simple tools. Writing such records as json text to simple text
files is about as interoperable and resiliant as you can get. At
the same time this library will build an index of an arbitrary set
of fields in your records so that queries can be run on the data
set as if it was in a database. The index can be freely deleted,
and if it becomes corrupted, it can simply be rebuilt, the core
data is never touched.

This is not exactly a full database, since it doesn't support
modification of records efficiently. If you want to change an
existing record, you can just do that, you can even open the file
in emacs and just edit it. However in that case the entire index
will be invalidated and rebuilt, which can take some
time. Therefore this should be considered an append only database,
since only append is implemented efficiently (which for our use
case is perfectly fine).
