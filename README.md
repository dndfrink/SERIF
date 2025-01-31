# SERIF: A Scalable, Extendable Real-Time Inference Framework

SERIF is a distributed inference serving framework that specifically targets real-time inference applications. One motivating paper for this framework is SneakPeek (citation TBD) - at a high level, the idea behind this paper is that there may be ways to optimize standard scheduling algorithms by taking a "sneak peek" at the data to be scheduled - which in this case, means pre-processing the data with a lightweight classifier and giving the scheduler hints based on the results from the classifier.

The framework implementation used to evaluate the algorithms has limitations that make it difficult to use for a wider variety of tasks or in a more real-world system configuration. For example, it only has the ability to execute on a single node - meaning data generation/parsing, scheduling, and model execution cannot be distributed, and configuring new models, data parsers, and schedulers involved having to extensively modify the framework code. This project aims to improve on the original system by adding a few new capabilities:

- Distributed Execution: Any amount of executor nodes can be configured, each of which can have any arbitrary amount of executor processes running
- Distributed KV Store: Redis is used as a shared KV implementation between nodes. Redis is configured completely independently of the executor nodes. This distributed KV store is supplemented by per-node shared memory caches that can be taken advantage of if executor nodes and request generation nodes are colocated.
- Extensible Configuration: This README provides extensive documentation surrounding the configuration of the system. Two different types of configurations are used:
    - System Configuration: Defines framework-wide system variables, executor nodes, Redis nodes, and input data stream locations
    - Application Configuration: Defines application-specific variables. Each application has a set of inference models that can be used by the scheduler to make decisions.

## Framework Configuration Documentation

The goal of this framework is to be as easily extensible as possible for new use cases. This section will provide details on how the configuration files and implementation modules can be used/modified to configure new applications within the framework.

### System Configuration

An example system configuration JSON file is provided at `src/configs/systemconfig.json`. Usage for each field is documented here:

- `seed` : Used for seeding RNG libraries. If this is 0, RNG will be seeded with the current time, otherwise, the seed provided will be used.
- `duration-in-secs` : If 0, the system will run until shut down, otherwise, it will run for the specified amount of seconds and shut down after it reaches the time limit.
- `scheduler-sleep` : How long the scheduler should sleep before checking the input queue for more requests
- `data-aware` : Whether to make use of the "data-aware" hints provided by SneakPeek or another user-defined estimator.
- `scheduler-type` : Used for specifying the scheduler to be used. More information will be provided later on how to map the identifier provided here to a user-defined scheduler implementation.
- `compute-nodes` : A list of dictionaries, each of which defines a node that can inference requests can be dispatched to. Each node should have the following things defined:
    - `name` : Easy string identifier for the node
    - `ip-address` : IP address of the node
    - `server-port` : Port that the `executor_server` process should use to receive requests (more information on the `executor_server` module will be provided later)
- `redis-nodes` : A list of nodes that have been configured as nodes in a Redis cluster. More documentation on the expected Redis configuration will be provided later. Each entry in this list should have the following members:
    - `ip-address` : IP address of the node
    - `ports` : A list of ports on the node for which a `redis-server` instance has been started.
- `scheduler-node` : Defines the configuration for the scheduler process. Two entries should be defined:
    - `ip-address` : IP address of the node running the scheduler process
    - `port` : Port that the scheduler process is listening on

### Application Configuration

An example application configuration file is located at `src/configs/applicationconfig.json`. Usage for each field is documented here:

- `name` : Name of the application
- `sla-latency` : Service level agreement latency - ie, how quick inferences should be done for this application
- `sample-duration` : Defines how often data should be sampled/inferenced for this application. For example, a sample duration of 4 would mean that data is parsed/inferenced every 4 seconds.
- `utility-matrix` : Utility matrix for the application. Briefly, a utility matrix defines how good/bad a given inference result is on a scale from -1.0 - 1.0. For example, a true positive would likely have a higher utility (1.0), while a false negative might have a lower utility (-1.0).
- `prior` : Input for the SneakPeek data classifier, optional if data aware scheduling is not being used.
- `model-profiles` : A list of dictionaries, each of which defines a specific profile/implementation for a specific inference model. Each entry should have the following members:
    - `name` : Name used to identify the model
    - `use-profile` : Used for simulation purposes. If this is true, the executor will just sleep/produce a random inference result based on the specified model profile, if false, the executor will actually run the inference using the model weights specified in the profile.
    - `modality` : Used to specify the underlying data type that is used as input for this model. This should correspond to one of the `name` fields in the `data-modalities` list in the system configuration file.
    - `cmat` : Confusion matrix for the model
    - `latencies` : Contains inference and context switch latency for each executor in the system. Each entry should be defined as `"<computeNodeName> : [<inferenceLatency>, <contextSwitchLatency>]`
    - `model-weights` : File containing pre-trained weights for this model. This will be loaded into Redis/distributed to all of the executor nodes.
    - `redis-key` : Unique identifier for this model to be used by Redis when distributing model weights across executors.

### Data Configuration

The data configuration contains all of the information about what types of data are being sent through the system and where that data is processed from. The following entries should be defined in this configuration:

- `num-groups` : Data from different modalities can be grouped together under the same ID. The example use case identified by the SneakPeek paper was for use in a hospital setting. In this case, each patient being monitored would be a "group", allowing streams of data from each modality to be assigned to the correct patients.
- `data-modalities` : Contains all of the information necessary for identifying/reading input data. The framework expects each stream of data to be read from a file. The user has complete control over how the data files are parsed/modified over the lifetime of the serving framework. Each entry in this list should have the following members:
    - `name` : Name of the given modality.
    - `synthetic-shape` : In the case that the framework is using simulated data for a given stream, this defines the shape of an `numpy.ndarray` that should be randomly initialized and forwarded to the executors for a request.
    - `use-synthetic-data` : Whether random, synthetic data should be generated for this modality or whether the framework should expect real input data streams for this modality.
    - `cache-entry-size` : The maximum size in bytes for a request from this modality after it has been serialized.
    - `retrieval-latencies` : Should contain entries for each compute node. The key for each entry should be the specified name of the compute node as defined in the system configuration, and the value should be a profiled average latency for retrieving an entry of this modality from Redis for each executor node in the system.
- `servers` : Contains a list of all of the request generating nodes in the system. Each entry in the list should have the following entries:
    - `name` : Name of the request generation node
    - `ip-address` : IP address for this request generation node
    - `port` : Port that the request generation server process should listen on
    - `workers` : Number of subworkers that should be created by the master request generation task. All groups assigned to each request generation node will be divided up among each of the workers.
    - `cache-entries-per-modality`: Defines how many entries should be allocated in the shared memory cache on this server for each modality. Entries should be in the format `"<nameOfModality>" : <numberOfEntries>`, and a `default` entry can be created to cover all modalities that are not explicitly defined.
    - `streams` : Contains entries for each modality that has real data to be processed. Each entry in this list should be set up as `"nameOfModality" : { <entries> }` where each entry in `entries` is defined as `"<groupId>" : "streamIdentifier"`, where this stream identifier is a filename, or some other identifier that the user-defined data parser can use to retrieve input data for this specific group.


For all configuration files, examples can be found in `src/configs/datatest`.

## Extending the framework

SERIF is designed to allow for custom models, custom data input parsing, and custom scheduling types. This section documents the necessary steps in order to add each of these components/allow the framework to utilize them.

### Models

The first step for adding a model is specifying its profile in the application configuration. The details of the configurations are listed above. After the model has been added to the configuration, it will be available for the scheduler to use, however, additional steps need to be taken to allow the executors to load its specified model weights and forward queries to it. In the `src/model_impl/__init__.py` file, the abstract base class `ModelImpl` is defined. All model types must be derived from this class. `ModelImpl` forces the class to implement `load()`, which is called during executor startup, and `forward()`, which is used when an inference request is directed to that model type. An example model implementation class can be found at `src/model_impl/x3d.py`. In addition, the `get_impl_for_model` function in `src/get_impl.py` must import the user-defined model implementation class and return an instance of the implementation class when the name of the input model matches the name of the model defined in the application configuration.

### Data Modalities

The first step for adding a data modality is specifying its details in the system configuration file. The details of this configuration are listed above. After the modality has been correctly specified, the data task will attempt to read data from each of the streams at the rate specified for the applications that make use of that modality (`sample-duration` in the application configuration file). In order to be able to actually read data from the input file, the user must implement a class that derives from the `DataImpl` class which is defined in `src/data_impl/__init__.py`. This classes forces derived classes to implement a `read()` method, which reads data from a file and returns an ndarray in the format expected by the backend models. An example data implementation can be found in `src/data_impl/readers.py`. In addition, the `get_reader_for_modality` function in `src/get_impl.py` must import the new user defined data reader class and return an instance of it when the input data modality matches the name specified for this modality in the system configuration.

### Custom Schedulers

The scheduling code used to evaluate this repo is found in the `sneakpeek` folder. Example scheduling algorithms are defined in the `sneakpeek/scheduling` directory. Each scheduling function must have a function signature defined by the `Scheduler` type (found in `src/scheduler_types.py`). New scheduling algorithms can be imported in `src/get_impl.py`, and the `get_schedule_impl` must return the correct user-defined scheduler function depending on the `scheduler-type` field in the system configuration.

## Redis Setup

This framework makes use of the Redis Cluster framework. This allows for the use of multiple processes across multiple systems to maintain replication of data. The Redis DB can be setup either on the same system as the executor nodes or across different systems. A minimum of 6 redis nodes set up in a cluster where data replication is enabled is recommended, however, the user could also specify only a single node if desired. All the framework needs to know is the IPs/ports of the `redis_server` processes that are in the cluster. An example redis startup script is provided in `start_redis.sh`.

## Example Usage

This is an example of how the system would be started on a single node using one executor and one request generation node.

First, create a virtual environment and install all required packages. Required python packages for the framework can be installed via the requirements.txt file, however, user-specific data/model implementations may require extra package installations.

Then, start up the Redis cluster.

`./start_redis.sh`

Then, in 3 other shells:

`python3 -m src.executor_server localhost <systemConfigPath> <applicationConfigPath> <dataConfigPath>`

`python3 -m src.data_server localhost <systemConfigPath> <applicationConfigPath> <dataConfigPath>`

`python3 -m src.scheduler <systemConfigPath> <dataConfigPath>`

In another shell:

`python3 -m src.pipeline_entrypoint <systemConfigPath> <applicationConfigPath> <dataConfigPath>`

This will then kick off the executors and request generation nodes, and the system will begin to function.
