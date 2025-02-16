# phasic

## Bootstrapping with CloudFormation

```sh
$ aws cloudformation deploy --capabilities CAPABILITY_IAM --stack-name phasic-bootstrap --template-file cloudformation/bootstrap.yaml
$ stack_role="$(aws cloudformation describe-stacks --output text --query 'Stacks[*].Outputs[?OutputKey==`"StackRole"`].OutputValue' --stack-name phasic-bootstrap)"
$ aws cloudformation deploy --capabilities CAPABILITY_IAM --parameter-overrides UserProfileName=zobar --role-arn "$stack_role" --stack-name phasic --template-file cloudformation/phasic.yaml
```

## Deleting

Creating a SageMaker domain also creates an EFS filesystem and some security groups, which are not cleaned up when the domain is destroyed.
