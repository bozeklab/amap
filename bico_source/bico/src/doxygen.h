/** 
 * @mainpage
 *
 * BICO [1] is a data stream algorithm for the k-means problem that combines
 * the data structure of the SIGMOD Test of Time award winning algorithm
 * BIRCH [2] with the theoretical concept of coresets for clustering problems.
 * It is implemented within the clustering library  CluE.
 * This documentation belongs to an extract from CluE with all files
 * necessary to compile and run BICO. The algorithm itself is implemented
 * in Bico / bico.h.
 *
 * - If you are interested in **using BICO** have a look at the
 *   [example](@ref p_bico_example)
 * - If you are interested in **understanding BICO** have a look at
 *   the [BICO website](http://ls2-www.cs.uni-dortmund.de/bico/) and [1]
 * .
 *
 * References
 * ----------
 * 1. Hendrik Fichtenberger, Marc Gillé, Melanie Schmidt, Chris Schwiegelshohn,
 *    Christian Sohler: BICO: BIRCH Meets Coresets for k-Means Clustering.
 *    ESA 2013: 481-492
 * 2. Tian Zhang, Raghu Ramakrishnan, Miron Livny:
 *    BIRCH: A New Data Clustering Algorithm and Its Applications.
 *    Data Min. Knowl. Discov. 1(2): 141-182 (1997)
 *
 * @page p_bico_example BICO example
 *
 * This is the source of quickstart.cpp. BICO is implemented in Bico / bico.h.
 * Prebuilt quickstart binaries can be downloaded from the
 * [BICO website](http://ls2-www.cs.uni-dortmund.de/bico/).
 * @include quickstart.cpp
 *
 * @page p_clue_rules CluE implementation
 *
 * This page lists several implementation details of CluE.
 * An example of how to use BICO can be found [here](@ref p_bico_example).
 *
 * @par Naming conventions
 * Suffixes used to group classes:
 * - ...Provider \n
 *   Abstract base class. Classes \em providing funcationality derive from \em Providers. Example:
 *   - ProxyProvider
 *   - SolutionProvider
 *   .
 * - ...Solution \n
 *   Data structure. Encapsulated computation \em solutions returned by algorithms. Example:
 *   - PartitionSolution
 *   - DoubleSolution
 *   .
 * .
 *
 * @par Required lifetime guarantees for paramters passed to CluE classes
 * An object passed to a non-const method (e.g. setter) as a pointer will
 * be copied if necessary, unless it is an input object
 * (e.g. points, proxies, ...) and except where otherwise stated.\n
 * Examples:
 * - <em>setMeasure(DissimilarityMeasure<T> const *measure)</em> will copy
 *   \em measure.
 * - <em>setInput(vector<T> const *proxies)</em> will propably not copy
 *   \em proxies or any \em T object.
 * - <em>setInput(vector<T*> const *input)</em> will propably not copy
 *   \em input or any \em T* pointer / \em T object.
 * .
 * In other words, when passed by a pointer
 * - objects containing logic will be copied if necessary,
 * - data structures will (propably) not.
 * .
 * This was decided on the basis that data structures might consume lots of
 * memory.
 *
 * @par Cloning objects
 * Some classes provide a \em clone() method which should be used
 * when copying objects passed by a pointer to avoid slicing effects.\n
 * Example:
 * - <em>DissimilarityMeasure<T>* clone() const</em>
 * .
 *
 * @par Unboxing objects
 * Some classes provide a static \em toType() method to unbox pointers.\n
 * Example:
 * - <em>static ProxyProvider<T>* toProxyProvider(SolutionProvider* s)</em>
 * .
 *
 */
