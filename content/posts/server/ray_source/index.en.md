---
title: "Ray Source-code Reading"
date: 2022-05-12T11:51:31+08:00
categories: ["server"]
summary: "This article introduces the source code of ray(1.8.x+)."
---

## Summary

This article introduces the source code of **ray(1.8.x+)**.

## To be translated

Oh Sorry!

This blog has't been translated to English, please wait for a little while...

## 重点问题

- 函数信息如何序列化？
  答：使用cloudpickle序列化：详见下文**cloudpickle**

- 如何维护函数池并实现全局可用
  答：
  - 将函数信息序列化后放到redis处：详见下文`run_function_on_all_workers`
  - 在需要的时候取回来

- 如何处理用户参数（args和kwargs）？
  答：`[key1, value1, key2, value2]`以此类推；如果参数里有`objectRef`，则会先获取对应value然后取回。

- 如何处理嵌套函数（一个函数里调用另一个函数）
  答：通过序列化闭包直接完成

- 如何处理函数依赖（一个函数的参数为另一个函数的结果）
  答：即上文处理函数参数中`objectRef`的部分

- 如何打包运行时环境
  答：并没有打包，但可以通过`runtime_env`参数处理运行时环境，也可以传pip config等帮助初始化环境。

- 如何处理函数中用到的全局变量
  答：
  - 在序列化函数的时候，直接将全局变量作为`global_ref`并序列化到函数中，反序列化后直接拿到这个数值。详见下文**类及全局变量**
  - 同spark一样，ray的全局变量可读，但**写不安全**。

## ray.init

### `init(address: Optional[str] = None, *, ……)`

- 注意到第二个参数就是*，在此之后的参数将只能通过key-value形式传递
- init支持三种模式
  - `ray.init()`——本地运行，启动所有关联的进程（包括Redis, a raylet（local scheduler）, a plasma store, a plasma manager, and some workers）
  - `ray.init(address="localhost:6379")`——连接到一个本地的cluster
  - `ray.init(address="ray://123.45.67.89:10001")`——连接到一个remote ray集群
- `local_mode`参数可以支持让代码串行，便于debug
- `include_dashboard`参数为True可以让ray起一个本地的dashboard显示状态
- `_enable_object_reconstruction`可以让ray在分布式环境下，丢失object后重新运行获得它的task来再次获得object

- **重点X**：在第三种模式中，会启动一个`ray.client`，解析目前参数并传给它
- 初始化时还会尝试增加`file descriptor limit`，获取更大空间
- 如果传进了`runtime_env`，启动`ray.job_config.JobConfig()`并将runtime_env传给它
- 创建或连接初始结点
  - 在第一种模式中，会创建一个新集群，全局变量`_global_node`为头结点（传入head=True）
  - 在第二或者第三种模式中，全局变量`_global_node`为子结点（传入head=False），并且connect_only = True
- 之后，如果有`job_config`的话，会调用`working_dir_pkg.rewrite_runtime_env_uris`来重写URI
- **重点√**：调用`connect()`function正式连接，详见下文。
- 最后返回`dict(_global_node.address_info, node_id=global_worker.core_worker.get_current_node_id().hex())`

### def connect(node, ……)

- 连接当前node到`raylet`, `Plasma`, 和`Redis`
- 调用`node.create_redis_client()`创建 redis client作为worker的属性
- 调用`_initialize_global_state` lazy初始化global state（lazy即只是存储了redis的address和密码），并没有真的初始化
- 如果是`driver` mode的话，初始化一个jobid
- 检查当前结点ray、python版本和集群ray、python版本是否一致（需要后者在redis中有version数据），不一致会报错
- 填充field到`job_config`中
- **重点X**：使用当前参数，初始化一个coreworker——`worker.core_worker = ray._raylet.CoreWorker()`，并拿到`gcs_client`
- **重点X**：启动一个import thread——`import_thread.ImportThread`
- 根据mode，启动`listener thread`和`log thread`
- **重点X**：如果非交互模式（正常脚本模式），调用`worker.run_function_on_all_workers`封装一个匿名函数`sys.path.insert(1, script_directory)`传进去，从这里我们可以看出ray其实是直接把脚本copy到working_dir的，然后用函数meta信息和上下文直接import对应函数执行。该函数具体信息见ray.worker。
- 如果client模式，`job_config`没带有`working_dir`的，同上处理
- 之后将所有`worker.cached_functions_to_run`调用`run_function_on_all_workers`
- 最后加一个`tracing`来跟踪

## ray.remote

- 感想：ray的api设计还是很优雅的
- ray在调用函数如`func.remote`后，返回`ObjectRef`对象。
- 当ray.get()的时候，拿着`ObjectRef`去`ObjectStore`里面找，如果有就返回值，如果没有就阻塞直到返回值。

### def remote(*args, **kwargs)

- 可以支持function/actor，可以手动分配资源`@ray.remote(num_gpus=1, max_calls=1, num_returns=2)`
- 当actor对象被 delete后，会完成当前任务再杀死，如果很急着关闭，可以调用`ray.kill()`
- 首先拿到global worker（每个进程一个），然后进行判断，如果没有传参直接`make_decorator(worker=worker)(args[0])`，有传参则解析传参然后`make_decorator`并将参数传入。
- 目前支持的参数有

```py
valid_kwargs = [
    "num_returns", "num_cpus", "num_gpus", "memory", "object_store_memory",
    "resources", "accelerator_type", "max_calls", "max_restarts",
    "max_task_retries", "max_retries", "runtime_env", "retry_exceptions",
    "placement_group"
]
```

### def make_decorator(参数略)

- 用inspect判断是函数还是类，如果是函数检查参数，并返回`ray.remote_function.RemoteFunction`
- 如果是类，检查参数，并返回`ray.actor.make_actor()`

### class RemoteFunction

- `__init__()`：
  - 首先检查是否协程，目前不支持async def function
  - 对于传入的function，加上`_inject_tracing_into_function`包裹以获取函数trace
  - **重点√**：使用`ParsedRuntimeEnv`来parse本地pip/conda config文件，详细见下文
  - **重点√**：使用`ray._private.signature.extract_signature`来extract（类）函数的signature，实际为`inspect.signature`获取function的参数和默认值。
  - 供用户调用的`.remote`由`_remote_proxy`调用`_remote`生成

- `__call__()`：不允许直接调用，报错返回

- `options()`：给用户的调整参数的方式，会重新wrap一个`RemoteFunction`给用户

- `_remote`：
  - 首先判断是不是`client_mode`，如果是，调用`client_mode_convert_function`返回`ClientRemoteFunction` **重点X**
  - 如果不是，拿到process的global_worker并调用`worker.check_connected()`检查连接
  - 之后检查这个函数是否被exported到这个process（使用`_last_export_session_and_job`与`worker.current_session_and_job`作比较）。如果没有，**重点√**：使用ray自定义的**cloudpickle**序列化该函数，标记`_last_export_session_and_job`和`_function_descriptor`属性，并用`worker.function_actor_manager.export(self)`export。cloudpickle详见下文。
  - 整理参数，`check_placement_group_index`检查放置的group的index
  - 调用`resources_from_resource_arguments`整理所有参数为字典
  - **重点X**，拿到当前worker的`parent_env`，连同当前task的`parsed_env`调用`override_task_or_actor_runtime_env`获取`merged_env`
  - 之后，根据是否cross_language 处理参数（如`flatten_args`，将signature和新传入的参数处理为单个list，如`flatten_args([1, 2, 3], {"a": 4})`处理为`[None, 1, None, 2, None, 3, "a", 4]`，`None`为dummy，指没有keyword的positional param）
  - 最后，调用`worker.core_worker.submit_task`提交任务，拿到`object_refs`，流程结束。

- `ParsedRuntimeEnv`类
  - 主要是运行环境相关的处理和依赖，没有包括全局变量等，使用`json`进行序列化和反序列化
  - 目前支持的字段有：

    ```py
    known_fields: Set[str] = {
        "working_dir", # 指定worker的工作目录，可以是一个目录也可以是一个archive压缩包，会将archive压缩包解压到指定工作目录
        "conda", "pip",   # pip requirements.txt或conda yaml
        "uris",        # 所需uri路径(uri是一个zip包)
        "containers",  # 支持docker镜像
        "env_vars",    # 环境变量
        "excludes", "_ray_release", "_ray_commit", "_inject_current_ray", "plugins"
    }
    ```

## ray.worker

- 重要属性
  - `node (ray.node.Node)`：worker所在结点
  - `mode`: `SCRIPT_MODE, LOCAL_MODE, WORKER_MODE`中的一个
  - `cached_functions_to_run (List)`：worker要运行的函数列表
  - `memory_monitor`: `memory_monitor.MemoryMonitor()`——在低内存环境处理报错信息
  - `function_actor_manager`：**重点X** 管理functions/actors的export/load
- 感想：类设计也很优雅，@property处理对外对内的变量，便于调用和维护

### worker.run_function_on_all_workers

- 注意这个函数并非我们直接的函数，而是封装过只接受一个worker info的参数。
- 如果还没`ray.init()`，就加入到cache里
- 如果已经init了，那就序列化（ray自带的`cloudpickle`）这个函数
- **重点X**：直接在driver尝试运行这个函数，`function({"worker": self})`，以校验这个函数是否正常。（注释写明：如果一个有问题的函数，最好在被exported之前被发现）
- `self.redis_client.setnx(b"Lock:" + key, 1)`尝试set（set if not exist），如果返回1，则加key成功，如果返回0，则没有加key成功（已经被用过了）来判断函数是否已经exported。

例子：

```bash
redis> SETNX mykey "Hello"
(integer) 1
redis> SETNX mykey "World"
(integer) 0
redis> GET mykey
"Hello"
```

- **重点√**：`check_oversized_function()`校验序列化函数是否太大，如果太大发送一个warning。
- **重点√**：使用redis的hset（设置哈希表键值），保存相关信息，随后rpush（推送到队列尾）。

```py
self.redis_client.hset(
                key,
                mapping={
                    "job_id": self.current_job_id.binary(),
                    "function_id": function_to_run_id,
                    "function": pickled_function,
                    "run_on_other_drivers": str(run_on_other_drivers),
                })
```

## cloudpickle——重写原生pickle

- 目的：序列化lambda和嵌套函数；正确处理`main`module；处理其他不能被序列化的objects
- 没有反序列化部分，这一部分用原生pickle做
- **非常非常厉害的序列化工具**：可以直接序列化类、类变量和类方法，全局变量也直接被序列化，甚至能够直接嵌套序列化，将用到的东西一起序列化，形成完整的package。

### CloudPickler

- 首先定义一个`dispatch_table`确定各个类型的处理方式如`classmethod`使用`_classmethod_reduce`处理，注意此处没有包括`function`
- 根据`pickle.HIGHEST_PROTOCOL`版本不同，有两种行为模式
  - `pickle.HIGHEST_PROTOCOL`>=5
    - 在初始化父类`Pickler`的时候，会多传入一个`buffer_callback`参数（`buffer_callback`可以让我们取回原始对象，请参考下面例子）

      ```py
      b = ZeroCopyByteArray(b"abc")
      buffers = []
      data = pickle.dumps(b, protocol=5)
      new_b = pickle.loads(data)
      print(b == new_b)  # True
      print(b is new_b)  # False: a copy was made
      data = pickle.dumps(b, protocol=5, buffer_callback=buffers.append)
      new_b = pickle.loads(data, buffers=buffers)
      print(b == new_b)  # True
      print(b is new_b)  # True: no copy was made
      ```

    - **重点X**：`reducer_override`——优先级比dispatch table高，可以突破type-specific的限制（如exception），同时基于`C _pickle.Pickler`效率更高。

  - 其他情况（`pickle.HIGHEST_PROTOCOL`<=4）
    - 初始化父类的时候不会传入`buffer_callback`
    - dispatch_table中加入`save_global`：可处理`types`
    - dispatch[types.FunctionType]加入`save_function`：可处理函数
- 等待编辑

### 实战——函数

我们以这个函数作为序列化的对象：

```py
def hello_world(name: str, age: int):
    print(f'hello world, {name} at {age}!')
```

- 用原生pickle序列化结果如下：**只有module和函数名**，反序列化后需要能import进这个函数才行。

```b
b'\x80\x03c__main__\nhello_world\nq\x00.'
```

- 用ray序列化结果如下，中间包含了**参数信息和函数体**，可以直接被其他地方反序列化并调用

```b
b'\x80\x05\x95@\x02\x00\x00\x00\x00\x00\x00\x8c\x1bray.cloudpickle.cloudpickle\x94\x8c\r_builtin_type\x94\x93\x94\x8c\nLambdaType\x94\x85\x94R\x94(h\x02\x8c\x08CodeType\x94\x85\x94R\x94(K\x02K\x00K\x02K\x06KCC\x1at\x00d\x01|\x00\x9b\x00d\x02|\x01\x9b\x00d\x03\x9d\x05\x83\x01\x01\x00d\x00S\x00\x94(N\x8c\rhello world, \x94\x8c\x04 at \x94\x8c\x01!\x94t\x94\x8c\x05print\x94\x85\x94\x8c\x04name\x94\x8c\x03age\x94\x86\x94\x8c%c:\\Users\\Peter\\Desktop\\test_pickle.py\x94\x8c\x0bhello_world\x94K\x05C\x02\x00\x01\x94))t\x94R\x94}\x94(\x8c\x0b__package__\x94\x8c\x00\x94\x8c\x08__name__\x94\x8c\x08__main__\x94\x8c\x08__file__\x94h\x13uNNNt\x94R\x94\x8c ray.cloudpickle.cloudpickle_fast\x94\x8c\x12_function_setstate\x94\x93\x94h\x1f}\x94}\x94(h\x1bh\x14\x8c\x0c__qualname__\x94h\x14\x8c\x0f__annotations__\x94}\x94(h\x10\x8c\x08builtins\x94\x8c\x03str\x94\x93\x94h\x11h(\x8c\x03int\x94\x93\x94u\x8c\x0e__kwdefaults__\x94N\x8c\x0c__defaults__\x94N\x8c\n__module__\x94h\x1c\x8c\x07__doc__\x94N\x8c\x0b__closure__\x94N\x8c\x17_cloudpickle_submodules\x94]\x94\x8c\x0b__globals__\x94}\x94u\x86\x94\x86R0.'
```

### 实战——类及全局变量

以这个类作为序列化的对象

```py
from ray import cloudpickle as pickle

global_value = 10086

class TestPickle:

    def __init__(self) -> None:
        self.variable = 123

    def hello_world(self, name: str, age: int):
        print(f'hello world, {name} at {age}!')
        print(self.variable)
        print(global_value)
    

with open('temp.pickle', 'wb') as f:
    pickle.dump(TestPickle, f)
```

- 使用ray的cloudpickle序列化后，可以直接读取并执行函数

```py
from ray import cloudpickle as pickle

with open('temp.pickle', 'rb') as f:
    cls = pickle.load(f)

c = cls()
c.hello_world('peter', 246)
```

### 实战——嵌套函数/嵌套类

ray可以将依赖的函数/依赖的类一起序列化，形成一个package，打包了整个环境。

```py
from ray import cloudpickle as pickle

def inner_function():
    print('inner function!')

class Inner_class:

    def __init__(self) -> None:
        print('inner class!')
    
    @staticmethod
    def class_function():
        print("inner class's function!")
        inner_function()


global_value = 10086

class TestPickle:

    def __init__(self) -> None:
        self.variable = 123

    def hello_world(self, name: str, age: int):
        print(f'hello world, {name} at {age}!')
        print(self.variable)
        print(global_value)
        inner_function()
        c = Inner_class()
        c.class_function()

def outer_function():
    c = TestPickle()
    c.hello_world('peter', 22)

with open('temp.pickle', 'wb') as f:
    pickle.dump(outer_function, f)

print(pickle.dumps(outer_function))

```

删除源代码（不然会优先import源码）反序列化后再执行

```py
with open('temp.pickle', 'rb') as f:
    func = pickle.load(f)

func()
'''输出
hello world, peter at 22!
123
10086
inner function!
inner class!
inner class's function!
inner function!
'''
```
