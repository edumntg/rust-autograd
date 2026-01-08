use std::cell::RefCell;
use std::rc::Rc;
use std::ops::Add;
use std::ops::Sub;
use std::fmt;

struct ValueData {
    data: f64,
    grad: f64,
    _backward: Option<Box<dyn FnMut()>>,
    _prev: Vec<Value>,
    _op: String,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(data: f64) -> Value {
        Value(Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            _backward: None,
            _prev: vec![],
            _op: String::new(),
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Value {
        let out_data = self.data() + rhs.data();

        // Create a new node with new value
        let out = Value(Rc::new(RefCell::new(ValueData {
            data: out_data,
            grad: 0.0,
            _backward: None,
            _prev: vec![self.clone(), rhs.clone()], // Track history
            _op: "+".to_string(),
        })));

        // Define backward pass closure
        let self_clone = self.clone();
        let rhs_clone = rhs.clone();
        let out_clone = out.clone();

        // Note: We use interior mutability (borrow_mut) to update gradients
        out.0.borrow_mut()._backward = Some(Box::new(move || {
            let out_grad = out_clone.grad();

            // self.grad += 1.0 * out.grad
            self_clone.0.borrow_mut().grad += out_grad;

            // rhs.grad += 1.0 * out.grad
            rhs_clone.0.borrow_mut().grad += out_grad;
        }));

        out
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Value) -> Value {
        let out_data = self.data() - rhs.data();

        // Create a new node with new value
        let out = Value(Rc::new(RefCell::new(ValueData {
            data: out_data,
            grad: 0.0,
            _backward: None,
            _prev: vec![self.clone(), rhs.clone()], // Track history
            _op: "-".to_string(),
        })));

        // Define backward pass closure
        let self_clone = self.clone();
        let rhs_clone = rhs.clone();
        let out_clone = out.clone();

        // Note: We use interior mutability (borrow_mut) to update gradients
        out.0.borrow_mut()._backward = Some(Box::new(move || {
            let out_grad = out_clone.grad();

            // self.grad -= 1.0 * out.grad
            self_clone.0.borrow_mut().grad -= out_grad;

            // rhs.grad -= 1.0 * out.grad
            rhs_clone.0.borrow_mut().grad -= out_grad;
        }));

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_new() {
        let v = Value::new(3.14);
        assert_eq!(v.data(), 3.14);
        assert_eq!(v.grad(), 0.0);
    }

    #[test]
    fn test_value_data() {
        let v = Value::new(2.71);
        assert_eq!(v.data(), 2.71);
    }

    #[test]
    fn test_value_grad() {
        let v = Value::new(1.0);
        assert_eq!(v.grad(), 0.0);
    }

    #[test]
    fn test_value_add() {
        let v1 = Value::new(1.0);
        let v2 = Value::new(2.0);

        let v_result = v1.add(v2);
        assert_eq!(v_result.data(), 3.0);
        assert_eq!(v_result.grad(), 0.0);
    }

    #[test]
    fn test_value_sub() {
        let v1 = Value::new(-3.0);
        let v2 = Value::new(-5.0);

        let v_result = v1.sub(v2); // -3-(-5) = -3+5=2
        assert_eq!(v_result.data(), 2.0);
        assert_eq!(v_result.grad(), 0.0);
    }
}