
from ruamel.yaml.comments import CommentedMap


def update_parameters(
        default_param_dict,
        adjusted_parameters,
        strict=False
):
    if not adjusted_parameters:
        return default_param_dict

    def merge(d, u):
        for k, v in u.items():
            if strict and k not in d:
                raise KeyError(f"Unknown parameter: {k}")

            # merge dicts recursively
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                merge(d[k], v)

            else:
                d[k] = v

                # copy comment if provided
                if isinstance(u, CommentedMap) and hasattr(u, "ca"):
                    if u.ca.items.get(k):
                        d.ca.items[k] = u.ca.items[k]

        return d

    return merge(default_param_dict, adjusted_parameters)


def create_default_parameter_file(
        filepath: str,
        default_param_dict: CommentedMap | dict | None = None,
        adjusted_parameters: CommentedMap | dict | None = None,
        strict: bool = False
):
    """
    Example usage:

    ```
    from ruamel.yaml import CommentedSeq

    default_params = CommentedMap(
        a=CommentedMap(
            b=1,
            c=True
        ),
        d=CommentedMap(
            e='some_string',
            f=CommentedSeq([1, 2, 3]),
            g=[1, 2, 3]
        ),
        e=5
    )
    default_params.yaml_set_comment_before_after_key('a', before='\nThis describes a')
    default_params.yaml_set_comment_before_after_key('d', before='\nThis describes d')
    default_params.yaml_set_comment_before_after_key('e', before='')
    default_params['d']['f'].fa.set_flow_style()

    adjusted_params = CommentedMap(
        a=CommentedMap(
            b=6
        ),
        e=7,
        # h='new'  # Only works in non-strict mode
    )
    adjusted_params['a'].yaml_add_eol_comment('<- this is important!', key='b')

    create_default_parameter_file(
        'parameters.yaml',
        default_param_dict=default_params,
        adjusted_parameters=adjusted_params,
        strict=True
    )
    ```
    """
    from ruamel.yaml import YAML
    yaml = YAML()

    final_param_dict = update_parameters(default_param_dict, adjusted_parameters, strict)

    with open(filepath, mode='w') as f:
        yaml.dump(final_param_dict, f)
        f.write('\n')


if __name__ == '__main__':

    from ruamel.yaml import CommentedSeq

    default_params = CommentedMap(
        a=CommentedMap(
            b=1,
            c=True
        ),
        d=CommentedMap(
            e='some_string',
            f=CommentedSeq([1, 2, 3]),
            g=[1, 2, 3]
        ),
        e=5
    )
    default_params.yaml_set_comment_before_after_key('a', before='\nThis describes a')
    default_params.yaml_set_comment_before_after_key('d', before='\nThis describes d')
    default_params.yaml_set_comment_before_after_key('e', before='')
    default_params['d']['f'].fa.set_flow_style()

    adjusted_params = CommentedMap(
        a=CommentedMap(
            b=6
        ),
        e=7,
        # h='new'  # Only works in non-strict mode
    )
    adjusted_params['a'].yaml_add_eol_comment('<- this is important!', key='b')

    create_default_parameter_file(
        '/media/julian/Data/tmp/default-params.yaml',
        default_param_dict=default_params,
        adjusted_parameters=adjusted_params,
        strict=True
    )
